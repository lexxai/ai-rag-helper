from asyncio import Queue, QueueFull


class EventBus:
    """Simple async pub/sub bus."""

    def __init__(self):
        self.listeners: set[Queue] = set()

    async def subscribe(self) -> Queue:
        q = Queue()
        self.listeners.add(q)
        return q

    async def unsubscribe(self, q: Queue):
        self.listeners.discard(q)

    async def publish(self, event: dict):
        for q in list(self.listeners):
            try:
                q.put_nowait(event)
            except QueueFull:
                pass
