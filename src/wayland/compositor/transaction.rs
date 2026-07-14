// The transaction model for handling surface states in Smithay
//
// The caching logic in `cache.rs` provides surfaces with a queue of
// pending states identified with numeric commit ids, allowing the compositor
// to precisely control *when* a state become active. This file is the second
// half: these identified states are grouped into transactions, which allow the
// synchronization of updates across surfaces.
//
// There are 2 main cases when the state of multiple surfaces must be updated
// atomically:
// - synchronized subsurface must have their state updated at the same time as their parents
// - The upcoming `wp_transaction` protocol
//
// In these situations, the individual states in a surface queue are grouped into a transaction
// and are all applied atomically when the transaction itself is applied. The logic for creating
// new transactions is currently the following:
//
// - Each surface has an implicit "pending" transaction, into which its newly committed state is
//   recorded
// - Furthermore, on commit, the pending transaction of all synchronized child subsurfaces is merged
//   into the current surface's pending transaction, and a new implicit transaction is started for those
//   children (logic is implemented in `handlers.rs`, in `PrivateSurfaceData::commit`).
// - Then, still on commit, if the surface is not a synchronized subsurface, its pending transaction is
//   directly applied
//
// This last step will change once we have support for explicit synchronization (and further in the future,
// of the wp_transaction protocol). Explicit synchronization introduces a notion of blockers: the transaction
// cannot be applied before all blockers are released, and thus must wait for it to be the case.
//
// For those situations, the (currently unused) `TransactionQueue` will come into play. It is a per-client
// queue of transactions, that stores and applies them by both respecting their topological order
// (ensuring that for each surface, states are applied in the correct order) and that all transactions
// wait before all their blockers are resolved to be merged. If a blocker is cancelled, the whole transaction
// it blocks is cancelled as well, and simply dropped. Thanks to the logic of `Cache::apply_state`, the
// associated state will be applied automatically when the next valid transaction is applied, ensuring
// global coherence.

// A significant part of the logic of this module is not yet used,
// but will be once proper transaction & blockers support is
// added to smithay
use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::{Arc, Mutex, atomic::AtomicBool},
};

use calloop::ping::Ping;
use wayland_server::{DisplayHandle, Resource, Weak, protocol::wl_surface::WlSurface};

use crate::utils::Serial;

use super::{CompositorHandler, add_blocker, tree::PrivateSurfaceData};

/// Kind for a [`Blocker`]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BlockerKind {
    /// A immediate blocker which needs to be cleared before all delayed blockers.
    Immediate,
    /// Defines a delayed blocker which will be evaluated after
    /// all immediate blockers are cleared.
    Delayed,
}

/// The transaction event a [`Blocker::notify`] call describes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotifyKind {
    /// All [`BlockerKind::Immediate`] blockers of the transaction have been cleared.
    ///
    /// The transaction is now only waiting for its [`BlockerKind::Delayed`] blockers.
    Ready,
    /// The transaction has been cancelled and its state will be discarded.
    ///
    /// The blocker will never be evaluated for this transaction again.
    Cancelled,
}

/// Context describing the transaction a [`Blocker::notify`] call originates from
#[derive(Debug)]
pub struct NotifyContext<'a> {
    /// The event that triggered this notification
    pub kind: NotifyKind,
    /// The surfaces taking part in the transaction.
    ///
    /// Note that due to transaction merging (e.g. synchronized subsurfaces)
    /// this can contain surfaces the blocker was never registered on.
    pub surfaces: &'a [WlSurface],
    /// All [`BlockerKind::Delayed`] blockers of the transaction, including the
    /// blocker being notified.
    pub delayed: &'a [Arc<dyn Blocker + Send + Sync>],
}

/// Types potentially blocking state changes
pub trait Blocker {
    /// Retrieve the current state of the blocker
    fn state(&self) -> BlockerState;

    /// Retrieve the kind of the blocker
    fn kind(&self) -> BlockerKind {
        BlockerKind::Immediate
    }

    /// Notifies a [`BlockerKind::Delayed`] blocker about a state change of a
    /// transaction it takes part in.
    ///
    /// This is called once per transaction when all [`BlockerKind::Immediate`]
    /// blockers have been cleared ([`NotifyKind::Ready`]), again whenever the set of
    /// pending [`BlockerKind::Delayed`] blockers of that transaction shrinks while the
    /// transaction is still pending, and once if the transaction is cancelled
    /// ([`NotifyKind::Cancelled`]).
    ///
    /// Notifications are delivered outside of any compositor transaction lock, so
    /// implementations are allowed to call
    /// [`CompositorClientState::blocker_cleared`](super::CompositorClientState::blocker_cleared)
    /// (directly or indirectly) from within `notify`.
    fn notify(&self, context: &NotifyContext<'_>) {
        let _ = context;
    }

    /// Returns `self` as [`std::any::Any`] for downcasting.
    ///
    /// [`BlockerKind::Delayed`] blockers that want to cooperate with other instances
    /// of themselves inside the same transaction (see [`NotifyContext::delayed`])
    /// should return `Some(self)` here.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}

/// States of a [`Blocker`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockerState {
    /// The block is pending and not resolved yet
    Pending,
    /// The block got released and changes can be applied
    Released,
    /// The block got cancelled and changes should be discarded
    Cancelled,
}

/// A simple [`Blocker`] barrier
#[derive(Debug, Clone)]
pub struct Barrier(Arc<AtomicBool>);

impl PartialEq for Barrier {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Barrier {}

impl Barrier {
    /// Initialize a new [`Barrier`] with the provided state
    pub fn new(signaled: bool) -> Self {
        Self(Arc::new(AtomicBool::new(signaled)))
    }

    /// Query if this barrier has been signaled
    #[inline]
    pub fn is_signaled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Signal this barrier
    #[inline]
    pub fn signal(&self) {
        self.0.store(true, std::sync::atomic::Ordering::Release)
    }
}

impl Blocker for Barrier {
    fn state(&self) -> BlockerState {
        if self.is_signaled() {
            BlockerState::Released
        } else {
            BlockerState::Pending
        }
    }
}

/// A barrier waiting for multiple surfaces to be notified
#[derive(Debug, Clone)]
pub struct SurfaceBarrier {
    barrier: Barrier,
    ping: Ping,
    ready: Arc<AtomicBool>,
    surfaces: Arc<Mutex<HashMap<Weak<WlSurface>, bool>>>,
}

impl SurfaceBarrier {
    /// Initialize an empty surface barrier
    ///
    /// [`Ping`] will be used to notify once all [`BlockerKind::Immediate`] blockers are cleared for tracked surfaces.
    /// On receiving the ping [`SurfaceBarrier::release`] has to be called to clear the surface barrier blockers and
    /// let the tracked surfaces make progress.
    pub fn new(ping: Ping) -> Self {
        Self::with_surfaces(ping, [])
    }

    /// Initializes a new [`SurfaceBarrier`] from a list of [`WlSurface`]
    ///
    /// See [`SurfaceBarrier::new`] for more information.
    pub fn with_surfaces<'a>(ping: Ping, surfaces: impl IntoIterator<Item = &'a WlSurface>) -> Self {
        let barrier = Barrier::new(false);
        let ready = Arc::new(AtomicBool::new(false));
        let barrier = Self {
            barrier,
            ping,
            ready,
            surfaces: Arc::new(Mutex::new(HashMap::new())),
        };
        barrier.register_surfaces(surfaces);
        barrier
    }

    /// Register surfaces to be tracked by this barrier.
    ///
    /// This will automatically insert a blocker that will resolve on [`SurfaceBarrier::release`].
    pub fn register_surfaces<'a>(&self, surfaces: impl IntoIterator<Item = &'a WlSurface>) {
        let mut lock = self.surfaces.lock().unwrap();
        for surface in surfaces {
            if let std::collections::hash_map::Entry::Vacant(vacant_entry) = lock.entry(surface.downgrade()) {
                vacant_entry.insert(false);
                add_blocker(
                    surface,
                    SurfaceBarrierBlocker {
                        barrier: self.barrier.clone(),
                        ping: self.ping.clone(),
                        ready: self.ready.clone(),
                        surfaces: self.surfaces.clone(),
                    },
                );
            }
        }
        self.ready.store(false, std::sync::atomic::Ordering::Release);
    }

    /// Release this barrier and clear the blocker on all tracked surfaces
    pub fn release<D: CompositorHandler + 'static>(&self, dh: &DisplayHandle, state: &mut D) {
        self.barrier.signal();
        #[allow(clippy::mutable_key_type)]
        let surfaces = { std::mem::take(&mut *self.surfaces.lock().unwrap()) };
        for (surface, _) in surfaces {
            let Ok(surface) = surface.upgrade() else {
                continue;
            };
            let Some(client) = surface.client() else {
                continue;
            };
            state.client_compositor_state(&client).blocker_cleared(state, dh);
        }
    }
}

#[derive(Debug, Clone)]
struct SurfaceBarrierBlocker {
    barrier: Barrier,
    ping: Ping,
    ready: Arc<AtomicBool>,
    surfaces: Arc<Mutex<HashMap<Weak<WlSurface>, bool>>>,
}

impl Blocker for SurfaceBarrierBlocker {
    fn state(&self) -> BlockerState {
        self.barrier.state()
    }

    fn kind(&self) -> BlockerKind {
        BlockerKind::Delayed
    }

    fn notify(&self, context: &NotifyContext<'_>) {
        if context.kind != NotifyKind::Ready {
            return;
        }
        let mut surfaces = self.surfaces.lock().unwrap();
        for surface in context.surfaces {
            surfaces
                .entry(surface.downgrade())
                .and_modify(|state| *state = true);
        }
        if surfaces
            .iter()
            .all(|(surface, state)| *state || !surface.is_alive())
        {
            let signaled = self.ready.swap(true, std::sync::atomic::Ordering::AcqRel);
            if !signaled {
                self.ping.ping();
            }
        }
    }
}

#[derive(Default)]
struct TransactionState {
    surfaces: Vec<(Weak<WlSurface>, Serial)>,
    blockers: Vec<Arc<dyn Blocker + Send + Sync>>,
}

impl TransactionState {
    fn insert(&mut self, surface: WlSurface, id: Serial) {
        if let Some(place) = self.surfaces.iter_mut().find(|place| place.0 == surface) {
            // the surface is already in the list, update the serial
            if place.1 < id {
                place.1 = id;
            }
        } else {
            // the surface is not in the list, insert it
            self.surfaces.push((surface.downgrade(), id));
        }
    }
}

enum TransactionInner {
    Data(TransactionState),
    Fused(Arc<Mutex<TransactionInner>>),
}

pub(crate) struct PendingTransaction {
    inner: Arc<Mutex<TransactionInner>>,
}

impl Default for PendingTransaction {
    fn default() -> Self {
        PendingTransaction {
            inner: Arc::new(Mutex::new(TransactionInner::Data(Default::default()))),
        }
    }
}

impl PendingTransaction {
    fn with_inner_state<T, F: FnOnce(&mut TransactionState) -> T>(&self, f: F) -> T {
        let mut next = self.inner.clone();
        loop {
            let tmp = match *next.lock().unwrap() {
                TransactionInner::Data(ref mut state) => return f(state),
                TransactionInner::Fused(ref into) => into.clone(),
            };
            next = tmp;
        }
    }

    pub(crate) fn insert_state(&self, surface: WlSurface, id: Serial) {
        self.with_inner_state(|state| state.insert(surface, id))
    }

    pub(crate) fn add_blocker<B: Blocker + Send + Sync + 'static>(&self, blocker: B) {
        self.with_inner_state(|state| state.blockers.push(Arc::new(blocker) as Arc<_>))
    }

    pub(crate) fn is_same_as(&self, other: &PendingTransaction) -> bool {
        let ptr1 = self.with_inner_state(|state| state as *const _);
        let ptr2 = other.with_inner_state(|state| state as *const _);
        ptr1 == ptr2
    }

    pub(crate) fn merge_into(&self, into: &PendingTransaction) {
        if self.is_same_as(into) {
            // nothing to do
            return;
        }
        // extract our pending surfaces and change our link
        let mut next = self.inner.clone();
        let my_state;
        loop {
            let tmp = {
                let mut guard = next.lock().unwrap();
                match *guard {
                    TransactionInner::Data(ref mut state) => {
                        my_state = std::mem::take(state);
                        *guard = TransactionInner::Fused(into.inner.clone());
                        break;
                    }
                    TransactionInner::Fused(ref into) => into.clone(),
                }
            };
            next = tmp;
        }
        // fuse our surfaces into our new transaction state
        self.with_inner_state(|state| {
            for (surface, id) in my_state.surfaces {
                if let Ok(surface) = surface.upgrade() {
                    state.insert(surface, id);
                }
            }
            state.blockers.extend(my_state.blockers);
        });
    }

    pub(crate) fn finalize(mut self) -> Transaction {
        // When finalizing a transaction, this *must* be the last handle to this transaction
        loop {
            let inner = match Arc::try_unwrap(self.inner) {
                Ok(mutex) => mutex.into_inner().unwrap(),
                Err(_) => panic!("Attempting to finalize a transaction but handle is not the last."),
            };
            match inner {
                TransactionInner::Data(TransactionState {
                    surfaces, blockers, ..
                }) => {
                    return Transaction {
                        surfaces,
                        blockers,
                        last_pending_delayed: None,
                    };
                }
                TransactionInner::Fused(into) => self.inner = into,
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct Transaction {
    surfaces: Vec<(Weak<WlSurface>, Serial)>,
    blockers: Vec<Arc<dyn Blocker + Send + Sync>>,
    // number of pending delayed blockers at the time of the last
    // delivered notification, `None` if never notified
    last_pending_delayed: Option<usize>,
}

/// A pending notification for the delayed blockers of a single transaction.
///
/// Built while holding the transaction queue lock, delivered after it
/// has been released, so that `notify` implementations may re-enter the
/// compositor (e.g. call `blocker_cleared`).
pub(crate) struct DelayedNotification {
    kind: NotifyKind,
    surfaces: Vec<WlSurface>,
    delayed: Vec<Arc<dyn Blocker + Send + Sync>>,
}

impl DelayedNotification {
    pub(crate) fn deliver(&self) {
        let context = NotifyContext {
            kind: self.kind,
            surfaces: &self.surfaces,
            delayed: &self.delayed,
        };
        for blocker in &self.delayed {
            blocker.notify(&context);
        }
    }
}

impl fmt::Debug for dyn Blocker + Send + Sync {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Blocker").field("state", &self.state()).finish()
    }
}

impl Transaction {
    /// Computes the global state of the transaction with regard to its blockers
    ///
    /// The logic is:
    ///
    /// - if at least one blocker is cancelled, the transaction is cancelled
    /// - otherwise, if at least one blocker is pending, the transaction is pending
    /// - otherwise, all blockers are released, and the transaction is also released
    pub(crate) fn state(&self, kind: Option<BlockerKind>) -> BlockerState {
        // In case all of our surfaces have been destroyed we can cancel this transaction
        // as we won't apply its state anyway
        if !self.surfaces.iter().any(|surface| surface.0.is_alive()) {
            return BlockerState::Cancelled;
        }

        use BlockerState::*;
        self.blockers
            .iter()
            .filter(|blocker| kind.map(|kind| kind == blocker.kind()).unwrap_or(true))
            .fold(Released, |acc, blocker| match (acc, blocker.state()) {
                (Cancelled, _) | (_, Cancelled) => Cancelled,
                (Pending, _) | (_, Pending) => Pending,
                (Released, Released) => Released,
            })
    }

    fn delayed_blockers(&self) -> impl Iterator<Item = &Arc<dyn Blocker + Send + Sync>> {
        self.blockers
            .iter()
            .filter(|blocker| blocker.kind() == BlockerKind::Delayed)
    }

    /// Builds a notification for the delayed blockers of this transaction,
    /// or `None` if it has no delayed blockers.
    fn notification(&self, kind: NotifyKind) -> Option<DelayedNotification> {
        let delayed: Vec<_> = self.delayed_blockers().cloned().collect();
        if delayed.is_empty() {
            return None;
        }
        let surfaces = self
            .surfaces
            .iter()
            .filter_map(|(surface, _)| surface.upgrade().ok())
            .collect();
        Some(DelayedNotification {
            kind,
            surfaces,
            delayed,
        })
    }

    pub(crate) fn apply<C: CompositorHandler + 'static>(self, dh: &DisplayHandle, state: &mut C) {
        for (surface, id) in self.surfaces {
            let Ok(surface) = surface.upgrade() else {
                continue;
            };

            PrivateSurfaceData::with_states(&surface, |states| {
                states.cached_state.apply_state(id, dh);
            });

            PrivateSurfaceData::invoke_post_commit_hooks::<C>(state, dh, &surface);

            tracing::trace!("Calling user implementation for wl_surface.commit");

            state.commit(&surface);
        }
    }
}

// This queue should be per-client
#[derive(Debug, Default)]
pub(crate) struct TransactionQueue {
    transactions: Vec<Transaction>,
    // we keep the hashset around to reuse allocations
    seen_surfaces: HashSet<u32>,
}

impl TransactionQueue {
    pub(crate) fn append(&mut self, t: Transaction) {
        self.transactions.push(t);
    }

    /// Removes and returns all transactions that are ready to be applied.
    ///
    /// Additionally returns the notifications for delayed blockers that became
    /// due during this pass. They *must* be delivered (see
    /// [`DelayedNotification::deliver`]) by the caller after all locks
    /// protecting this queue have been released.
    pub(crate) fn take_ready(&mut self) -> (Vec<Transaction>, Vec<DelayedNotification>) {
        // FIXME: Get rid of this allocation here
        let mut ready_transactions = Vec::new();
        let mut notifications = Vec::new();
        // this is a very non-optimized implementation
        // we just iterate over the queue of transactions, keeping track of which
        // surface we have seen as they encode transaction dependencies
        self.seen_surfaces.clear();
        // manually iterate as we're going to modify the Vec while iterating on it
        let mut i = 0;
        // the loop will terminate, as at every iteration either i is incremented by 1
        // or the length of self.transactions is reduced by 1.
        while i < self.transactions.len() {
            let mut skip = false;
            // does the transaction have any active blocker?
            match self.transactions[i].state(Some(BlockerKind::Immediate)) {
                BlockerState::Cancelled => {
                    // this transaction is cancelled, remove it without further processing,
                    // but let its delayed blockers know they will never be evaluated again
                    let transaction = self.transactions.remove(i);
                    notifications.extend(transaction.notification(NotifyKind::Cancelled));
                    continue;
                }
                BlockerState::Pending => {
                    skip = true;
                }
                BlockerState::Released => {}
            }
            // if not, does this transaction depend on any previous transaction?
            if !skip {
                for (s, _) in &self.transactions[i].surfaces {
                    // TODO: is this alive check still needed?
                    if !s.is_alive() {
                        continue;
                    }
                    if self.seen_surfaces.contains(&s.id().protocol_id()) {
                        skip = true;
                        break;
                    }
                }
            }

            if skip {
                // this transaction is not yet ready and should be skipped, add its surfaces to our
                // seen list
                for (s, _) in &self.transactions[i].surfaces {
                    // TODO: is this alive check still needed?
                    if !s.is_alive() {
                        continue;
                    }
                    self.seen_surfaces.insert(s.id().protocol_id());
                }
                i += 1;
            } else {
                // all immediate blockers are cleared, check the delayed ones
                let transaction = &mut self.transactions[i];

                match transaction.state(None) {
                    BlockerState::Pending => {
                        // still waiting for delayed blockers; notify them, but only
                        // when their observable state changed since the last pass to
                        // avoid notifying over and over
                        let pending_delayed = transaction
                            .delayed_blockers()
                            .filter(|blocker| blocker.state() == BlockerState::Pending)
                            .count();
                        if transaction.last_pending_delayed != Some(pending_delayed) {
                            transaction.last_pending_delayed = Some(pending_delayed);
                            notifications.extend(transaction.notification(NotifyKind::Ready));
                        }
                        // this transaction is not yet ready and should be skipped, add its surfaces to our
                        // seen list
                        for (s, _) in &transaction.surfaces {
                            // TODO: is this alive check still needed?
                            if !s.is_alive() {
                                continue;
                            }
                            self.seen_surfaces.insert(s.id().protocol_id());
                        }
                        i += 1;
                    }
                    BlockerState::Released => ready_transactions.push(self.transactions.remove(i)),
                    BlockerState::Cancelled => {
                        // this transaction is cancelled, remove it without further processing,
                        // but let its delayed blockers know they will never be evaluated again
                        let transaction = self.transactions.remove(i);
                        notifications.extend(transaction.notification(NotifyKind::Cancelled));
                    }
                }
            }
        }

        (ready_transactions, notifications)
    }
}
