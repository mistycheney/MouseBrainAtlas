
class SignalEmitter(QObject):
    moved = pyqtSignal(int, int, int, int)
    clicked = pyqtSignal()
    released = pyqtSignal()
    
    def __init__(self, parent):
        super(SignalEmitter, self).__init__()
        self.parent = parent


class QGraphicsPathItemModified(QGraphicsPathItem):

	def __init__(self, parent=None, gui=None, *args, **kwargs):
		super(self.__class__, self).__init__(parent, *args, **kwargs)
		self.signal_emitter = SignalEmitter(parent=self)
		self.just_created = True # this flag is used to make sure a click is not emitted right after item creation
								# basically, ignore the first press and release event
		self.gui = gui

	def mousePressEvent(self, event):
		if not self.just_created:
			print self, 'received mousePressEvent'

			self.press_scene_x = event.scenePos().x()
			self.press_scene_y = event.scenePos().y()

			self.center_scene_x_before_move = self.scenePos().x()
			self.center_scene_y_before_move = self.scenePos().y()

			self.gui.selected_polygon = self

			QGraphicsPathItem.mousePressEvent(self, event)
			self.signal_emitter.clicked.emit()

			if 'labelTextArtist' in self.gui.accepted_proposals[self.gui.selected_polygon]:
				label_pos_before_move = self.gui.accepted_proposals[self.gui.selected_polygon]['labelTextArtist'].scenePos()
				self.label_pos_before_move_x = label_pos_before_move.x()
				self.label_pos_before_move_y = label_pos_before_move.y()

			print self.label_pos_before_move_x

		self.just_created = False

	def mouseReleaseEvent(self, event):
		if not self.just_created:
			print self, 'received mouseReleaseEvent'
			
			release_scene_pos = event.scenePos()
			self.release_scene_x = release_scene_pos.x()
			self.release_scene_y = release_scene_pos.y()

			QGraphicsPathItem.mouseReleaseEvent(self, event)
			self.signal_emitter.released.emit()

			self.press_scene_x = None
			self.press_scene_y = None

			self.center_scene_x_before_move = None
			self.center_scene_y_before_move = None

	def mouseMoveEvent(self, event):
		print self, 'received mouseMoveEvent'
		self.signal_emitter.moved.emit(event.scenePos().x(), event.scenePos().y(), self.press_scene_x, self.press_scene_y)

		if not self.gui.mode == Mode.IDLE:
			QGraphicsPathItem.mouseMoveEvent(self, event)


class QGraphicsEllipseItemModified(QGraphicsEllipseItem):

	def __init__(self, parent=None, *args, **kwargs):
		super(self.__class__, self).__init__(parent, *args, **kwargs)
		self.signal_emitter = SignalEmitter(parent=self)
		self.just_created = True # this flag is used to make sure a click is not emitted right after item creation
								# basically, ignore the first press and release event

	def mousePressEvent(self, event):
		if not self.just_created:
			print self, 'received mousePressEvent'
			QGraphicsEllipseItem.mousePressEvent(self, event)
			self.signal_emitter.clicked.emit()

			self.press_scene_x = event.scenePos().x()
			self.press_scene_y = event.scenePos().y()

			self.center_scene_x_before_move = self.scenePos().x()
			self.center_scene_y_before_move = self.scenePos().y()

		self.just_created = False

	def mouseReleaseEvent(self, event):
		if not self.just_created:
			print self, 'received mouseReleaseEvent'
			QGraphicsEllipseItem.mouseReleaseEvent(self, event)
			self.signal_emitter.released.emit()

			self.press_scene_x = None
			self.press_scene_y = None

			self.center_scene_x_before_move = None
			self.center_scene_y_before_move = None

	def mouseMoveEvent(self, event):
		print self, 'received mouseMoveEvent'
		self.signal_emitter.moved.emit(event.scenePos().x(), event.scenePos().y(), self.press_scene_x, self.press_scene_y)
		QGraphicsEllipseItem.mouseMoveEvent(self, event)
