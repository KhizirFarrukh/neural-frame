"""Before/after comparison widget: drag the divider across the frame."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget


class CompareView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._before: QPixmap | None = None
        self._after: QPixmap | None = None
        self.split = 0.5
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.SplitHCursor)
        self.setToolTip("Drag to compare: left = source (bicubic), right = NeuralFrame")

    def set_images(self, before: QPixmap, after: QPixmap) -> None:
        self._before, self._after = before, after
        self.update()

    def clear(self) -> None:
        self._before = self._after = None
        self.update()

    @property
    def has_images(self) -> bool:
        return self._before is not None and self._after is not None

    def sizeHint(self) -> QSize:
        return QSize(640, 360)

    def _target(self) -> QRectF:
        assert self._after is not None
        w, h = self._after.width(), self._after.height()
        scale = min(self.width() / max(w, 1), self.height() / max(h, 1))
        tw, th = w * scale, h * scale
        return QRectF((self.width() - tw) / 2, (self.height() - th) / 2, tw, th)

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(16, 18, 22))
        if self._before is None or self._after is None:
            painter.setPen(QColor(150, 155, 165))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "before / after appears here when done")
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        target = self._target()
        painter.drawPixmap(target, self._before, QRectF(self._before.rect()))
        x = target.left() + target.width() * self.split
        painter.save()
        painter.setClipRect(QRectF(QPointF(x, target.top()), target.bottomRight()))
        painter.drawPixmap(target, self._after, QRectF(self._after.rect()))
        painter.restore()
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(QPointF(x, target.top()), QPointF(x, target.bottom()))
        for text, left in (("before", True), ("after", False)):
            box = QRectF(0, 0, 56, 20)
            box.moveTop(target.top() + 8)
            if left:
                box.moveLeft(target.left() + 8)
            else:
                box.moveRight(target.right() - 8)
            painter.fillRect(box, QColor(0, 0, 0, 150))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _move(self, event: QMouseEvent) -> None:
        if self._after is None:
            return
        target = self._target()
        self.split = min(1.0, max(0.0, (event.position().x() - target.left()) / max(target.width(), 1.0)))
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._move(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._move(event)
