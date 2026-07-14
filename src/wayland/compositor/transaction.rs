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

type BarrierNotifier = Arc<dyn Fn() + Send + Sync>;

#[derive(Debug, Default)]
struct BarrierSurfaceEntry {
    // all immediate blockers of a transaction containing this surface have been cleared
    notified: bool,
    // that transaction also contains foreign pending delayed blockers we cannot fuse with
    foreign_pending: bool,
}

struct BarrierGroupState {
    #[allow(clippy::mutable_key_type)]
    surfaces: HashMap<Weak<WlSurface>, BarrierSurfaceEntry>,
    // released flag of every barrier fused into this group
    members: Vec<Arc<AtomicBool>>,
    // ready notifier of every barrier fused into this group
    notifiers: Vec<BarrierNotifier>,
    // the ready notification has been sent
    signaled: bool,
}

impl fmt::Debug for BarrierGroupState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BarrierGroupState")
            .field("surfaces", &self.surfaces)
            .field("members", &self.members)
            .field("signaled", &self.signaled)
            .finish_non_exhaustive()
    }
}

// Barrier groups form a union-find structure: fusing two groups redirects one
// of them to the other, mirroring how `PendingTransaction`s are fused when
// transactions are merged.
#[derive(Debug)]
enum BarrierGroupInner {
    State(BarrierGroupState),
    Fused(BarrierGroup),
}

#[derive(Debug, Clone)]
struct BarrierGroup(Arc<Mutex<BarrierGroupInner>>);

impl BarrierGroup {
    fn new(state: BarrierGroupState) -> Self {
        BarrierGroup(Arc::new(Mutex::new(BarrierGroupInner::State(state))))
    }

    /// Find the current root of this group
    fn root(&self) -> BarrierGroup {
        let mut next = self.clone();
        loop {
            let tmp = match &*next.0.lock().unwrap() {
                BarrierGroupInner::State(_) => None,
                BarrierGroupInner::Fused(into) => Some(into.clone()),
            };
            match tmp {
                Some(into) => next = into,
                None => return next,
            }
        }
    }

    fn with_state<T>(&self, f: impl FnOnce(&mut BarrierGroupState) -> T) -> T {
        let mut next = self.clone();
        loop {
            let tmp = {
                let mut guard = next.0.lock().unwrap();
                match &mut *guard {
                    BarrierGroupInner::State(state) => return f(state),
                    BarrierGroupInner::Fused(into) => into.clone(),
                }
            };
            next = tmp;
        }
    }

    /// Fuse two groups into one.
    ///
    /// Afterwards both groups share their surfaces, members and notifiers: the ready
    /// notification fires once the union of the tracked surfaces is ready and the
    /// blockers of all fused barriers resolve together once every member barrier
    /// has been released.
    fn fuse(&self, other: &BarrierGroup) {
        loop {
            let a = self.root();
            let b = other.root();
            if Arc::ptr_eq(&a.0, &b.0) {
                return;
            }
            // lock the two roots in address order to avoid lock cycles
            let (first, second) = if Arc::as_ptr(&a.0) < Arc::as_ptr(&b.0) {
                (a, b)
            } else {
                (b, a)
            };
            let mut g1 = first.0.lock().unwrap();
            let mut g2 = second.0.lock().unwrap();
            // if another fuse redirected one of the roots in the meantime, retry
            if !matches!(&*g1, BarrierGroupInner::State(_)) || !matches!(&*g2, BarrierGroupInner::State(_)) {
                continue;
            }
            let BarrierGroupInner::State(merged) =
                std::mem::replace(&mut *g2, BarrierGroupInner::Fused(first.clone()))
            else {
                unreachable!()
            };
            let BarrierGroupInner::State(state) = &mut *g1 else {
                unreachable!()
            };
            state.surfaces.extend(merged.surfaces);
            state.members.extend(merged.members);
            state.notifiers.extend(merged.notifiers);
            // the readiness condition changed, evaluate it anew
            state.signaled = false;
            return;
        }
    }

    /// Checks whether the group became ready and if so sends the (one-shot) ready
    /// notification to all fused barriers.
    fn maybe_signal(&self) {
        let notifiers = self.with_state(|state| {
            if state.signaled {
                return None;
            }
            let mut any_alive = false;
            let ready = state.surfaces.iter().all(|(surface, entry)| {
                if !surface.is_alive() {
                    // dead surfaces are excused, their state will never be applied
                    return true;
                }
                any_alive = true;
                entry.notified && !entry.foreign_pending
            });
            if ready && any_alive {
                state.signaled = true;
                Some(state.notifiers.clone())
            } else {
                None
            }
        });
        // invoke the notifiers without holding the group lock, so they may
        // call `SurfaceBarrier::release` synchronously
        if let Some(notifiers) = notifiers {
            for notifier in notifiers {
                notifier();
            }
        }
    }
}

/// A barrier synchronizing state application across multiple, possibly unrelated, surfaces.
///
/// A `SurfaceBarrier` places a [`BlockerKind::Delayed`] blocker on every registered
/// surface. Once every registered surface reached the point where all
/// [`BlockerKind::Immediate`] blockers of its transaction have been cleared — i.e. the
/// only thing holding back the state application is this barrier — the provided
/// notifier is invoked. Calling [`SurfaceBarrier::release`] then applies the pending
/// state of all registered surfaces.
///
/// One example use case are animation transactions spanning multiple otherwise
/// unrelated surfaces.
///
/// # Grouping and atomicity
///
/// Transactions can merge (e.g. through synchronized subsurfaces), so blockers of two
/// different `SurfaceBarrier`s can end up in the same transaction. In that case the
/// two barriers are *fused*: they behave like a single barrier spanning the union of
/// their surfaces. The ready notifiers of both barriers fire together (once the union
/// is ready) and no state is applied before *both* barriers have been released. This
/// keeps the application of the whole group atomic instead of letting one barrier
/// release a part of a transaction group while another part keeps waiting.
///
/// If a transaction contains a foreign pending [`BlockerKind::Delayed`] blocker (one
/// not belonging to a `SurfaceBarrier`), the ready notification is deferred until that
/// blocker resolves, so that when the notifier fires, releasing the barrier is
/// guaranteed to immediately apply the pending state of all registered surfaces.
///
/// # One-shot
///
/// A `SurfaceBarrier` is one-shot: the notifier fires at most once (unless a later
/// fuse re-arms the evaluation) and after [`SurfaceBarrier::release`] the barrier is
/// spent. Create a new `SurfaceBarrier` for the next synchronization point instead of
/// re-using a released one.
///
/// The notifier only fires while at least one tracked surface is alive. If every
/// tracked surface is destroyed (or their transactions cancelled) before the barrier
/// becomes ready, the notifier never fires — compositors should observe surface
/// destruction themselves and release or drop the barrier accordingly.
#[derive(Debug, Clone)]
pub struct SurfaceBarrier {
    group: BarrierGroup,
    released: Arc<AtomicBool>,
}

impl SurfaceBarrier {
    /// Initialize an empty surface barrier.
    ///
    /// `notifier` will be invoked once all [`BlockerKind::Immediate`] blockers are
    /// cleared for all tracked surfaces (see the type level documentation for the
    /// exact semantics). It is invoked outside of any compositor lock, directly from
    /// within the surface commit or
    /// [`blocker_cleared`](super::CompositorClientState::blocker_cleared) call that
    /// completed the condition, so [`SurfaceBarrier::release`] may be called from
    /// within it to apply the pending states within the same event loop iteration.
    /// Alternatively it can simply wake the event loop, e.g. via
    /// [`calloop::ping::Ping`]: `SurfaceBarrier::new(move || ping.ping())`.
    ///
    /// Note that the notifier can fire from within a surface commit dispatch; if it
    /// mutates compositor state it must do so through appropriately shared state.
    pub fn new(notifier: impl Fn() + Send + Sync + 'static) -> Self {
        Self::with_surfaces(notifier, [])
    }

    /// Initializes a new [`SurfaceBarrier`] from a list of [`WlSurface`]
    ///
    /// See [`SurfaceBarrier::new`] for more information.
    pub fn with_surfaces<'a>(
        notifier: impl Fn() + Send + Sync + 'static,
        surfaces: impl IntoIterator<Item = &'a WlSurface>,
    ) -> Self {
        let released = Arc::new(AtomicBool::new(false));
        let barrier = Self {
            group: BarrierGroup::new(BarrierGroupState {
                surfaces: HashMap::new(),
                members: vec![released.clone()],
                notifiers: vec![Arc::new(notifier)],
                signaled: false,
            }),
            released,
        };
        barrier.register_surfaces(surfaces);
        barrier
    }

    /// Register surfaces to be tracked by this barrier.
    ///
    /// This will automatically insert a blocker that will resolve on [`SurfaceBarrier::release`].
    ///
    /// The blocker attaches to the pending transaction of each surface, so this is
    /// meant to be called before the surface commit that should be held back, e.g.
    /// from within a pre-commit hook or before sending the configure the client is
    /// expected to ack. Registering additional surfaces re-arms the ready
    /// notification.
    pub fn register_surfaces<'a>(&self, surfaces: impl IntoIterator<Item = &'a WlSurface>) {
        let root = self.group.root();
        let mut new_surfaces = Vec::new();
        root.with_state(|state| {
            for surface in surfaces {
                if let std::collections::hash_map::Entry::Vacant(vacant_entry) =
                    state.surfaces.entry(surface.downgrade())
                {
                    vacant_entry.insert(BarrierSurfaceEntry::default());
                    state.signaled = false;
                    new_surfaces.push(surface.clone());
                }
            }
        });
        for surface in new_surfaces {
            add_blocker(&surface, SurfaceBarrierBlocker { group: root.clone() });
        }
    }

    /// Query if the ready notification has fired.
    ///
    /// This allows polling the barrier state instead of (or in addition to) using the
    /// notifier passed to [`SurfaceBarrier::new`].
    pub fn is_ready(&self) -> bool {
        self.group.with_state(|state| state.signaled)
    }

    /// Query if [`SurfaceBarrier::release`] has been called on this barrier.
    ///
    /// Note that this only reflects this barrier's own release, not that of
    /// other barriers fused with it.
    pub fn is_released(&self) -> bool {
        self.released.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Release this barrier and clear the blocker on all tracked surfaces.
    ///
    /// If this barrier has been fused with other barriers (see the type level
    /// documentation), the blockers only resolve once every fused barrier has been
    /// released; the last release applies the pending states of the whole group.
    pub fn release<D: CompositorHandler + 'static>(&self, dh: &DisplayHandle, state: &mut D) {
        self.released.store(true, std::sync::atomic::Ordering::Release);
        let surfaces: Vec<Weak<WlSurface>> = self.group.with_state(|group| {
            if !group
                .members
                .iter()
                .all(|released| released.load(std::sync::atomic::Ordering::Acquire))
            {
                // other barriers of the group are not released yet, the last
                // one will wake the surfaces
                return Vec::new();
            }
            group.surfaces.drain().map(|(surface, _)| surface).collect()
        });
        let mut seen_clients = HashSet::new();
        for surface in surfaces {
            let Ok(surface) = surface.upgrade() else {
                continue;
            };
            let Some(client) = surface.client() else {
                continue;
            };
            if seen_clients.insert(client.id()) {
                state.client_compositor_state(&client).blocker_cleared(state, dh);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct SurfaceBarrierBlocker {
    group: BarrierGroup,
}

impl Blocker for SurfaceBarrierBlocker {
    fn state(&self) -> BlockerState {
        let released = self.group.with_state(|state| {
            state
                .members
                .iter()
                .all(|released| released.load(std::sync::atomic::Ordering::Acquire))
        });
        if released {
            BlockerState::Released
        } else {
            BlockerState::Pending
        }
    }

    fn kind(&self) -> BlockerKind {
        BlockerKind::Delayed
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn notify(&self, context: &NotifyContext<'_>) {
        match context.kind {
            NotifyKind::Ready => {
                // fuse with all surface barriers that ended up in the same transaction,
                // so the whole group stays atomic (this includes ourselves, which is a no-op)
                for peer in context.delayed {
                    if let Some(peer) = peer
                        .as_any()
                        .and_then(|any| any.downcast_ref::<SurfaceBarrierBlocker>())
                    {
                        self.group.fuse(&peer.group);
                    }
                }
                // check for foreign delayed blockers we cannot coordinate with; defer
                // the ready notification until they resolved
                let foreign_pending = context.delayed.iter().any(|blocker| {
                    blocker
                        .as_any()
                        .map(|any| any.downcast_ref::<SurfaceBarrierBlocker>().is_none())
                        .unwrap_or(true)
                        && blocker.state() == BlockerState::Pending
                });
                self.group.with_state(|state| {
                    for surface in context.surfaces {
                        if let Some(entry) = state.surfaces.get_mut(&surface.downgrade()) {
                            entry.notified = true;
                            entry.foreign_pending = foreign_pending;
                        }
                    }
                });
            }
            NotifyKind::Cancelled => {
                // the transaction was cancelled: the pending state of its surfaces
                // will never be applied, waiting for them would stall the whole
                // group forever
                self.group.with_state(|state| {
                    for surface in context.surfaces {
                        state.surfaces.remove(&surface.downgrade());
                    }
                });
            }
        }
        self.group.maybe_signal();
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
