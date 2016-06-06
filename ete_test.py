# We will need to create Qt4 items
from PyQt4 import QtCore
from PyQt4.QtGui import QGraphicsRectItem, QGraphicsSimpleTextItem, \
    QColor, QPen, QBrush, QGraphicsRectItem

from ete3 import Tree, faces, TreeStyle, NodeStyle

# To play with random colors
import colorsys
import random

class InteractiveItem(QGraphicsRectItem):
    def __init__(self, *arg, **karg):
        QGraphicsRectItem.__init__(self, *arg, **karg)
        self.node = None
        self.color_rect = None
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def hoverEnterEvent (self, e):
        if self.color_rect is not None:
            self.color_rect.setBrush(QBrush(QColor(200, 100, 100, 100)))

    def hoverLeaveEvent(self, e):
        if self.color_rect is not None:
            self.color_rect.setBrush(QBrush(QColor(100, 100, 200, 100)))


    def mousePressEvent(self, e):
        print(self.node.name)
        print(self.node.dist)

def ugly_name_face(node, *args, **kargs):
    """ This is my item generator. It must receive a node object, and
    returns a Qt4 graphics item that can be used as a node face.
    """
    width = node.dist * 2.5
    height = 12
    masterItem = InteractiveItem(0, 0, width, height)
    masterItem.node = node
    masterItem.setPen(QPen(QtCore.Qt.NoPen))

    color_rect = QGraphicsRectItem(masterItem.rect())
    color_rect.setParentItem(masterItem)
    color_rect.setBrush(QBrush(QColor(100, 100, 200, 100)))
    color_rect.setPen(QPen(QtCore.Qt.NoPen))

    masterItem.color_rect = color_rect

    return masterItem

def master_ly(node):
    if not node.is_leaf() and not node.is_root():
        F = faces.DynamicItemFace(ugly_name_face, 50)
        faces.add_face_to_node(F, node, 0, position="float")

def get_example_tree():
    with open("avian_tree", "r") as infile:
        data = infile.read()

    t = Tree(data)

    ts = TreeStyle()
    ts.layout_fn = master_ly
    ts.title.add_face(faces.TextFace("Drawing your own Qt Faces", fsize=15), 0)
    return t, ts

def main():
    t, ts = get_example_tree()
    t.show(tree_style=ts)

if __name__ == "__main__":
    main()
