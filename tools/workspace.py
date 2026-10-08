from pathlib import Path


class Workspace:

    def __init__(self, root: str):
        self.root = Path(root).resolve()

    def resolve(self, path: str) -> Path:
        target = (self.root / path).resolve()

        try:
            target.relative_to(self.root)
        except ValueError:
            raise PermissionError(
                "Path is outside workspace"
            )

        return target