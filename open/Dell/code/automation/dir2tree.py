import os
import os.path as osp
from collections import defaultdict
from typing import List, Optional

class TreeNode(object):
    def __init__(self, name: str = '', path: str = '', childs: List["TreeNode"] = None):
        self.name = name
        self.path = path
        self.childs = childs

    def add_child(self, child: "TreeNode"):
        """
        Add child.

        Args:
            child: child to be added

        Returns:
            None
        """
        if not self.childs:     # self.childs = None
            self.childs = []
        self.childs.append(child)

    def update_tree(self):
        """
        Update tree with new added files.
        """
        if not osp.isdir(self.path) or not os.listdir(self.path):
            return

        # delete
        dir_list = os.listdir(self.path)
        if self.childs:
            for child in self.childs:
                if child.name not in dir_list:
                    self.childs.remove(child)

            # check, modify, add
            root_childs = [child.name for child in self.childs]
            for file in dir_list:
                if file in root_childs:
                    # check and modify
                    self.childs[root_childs.index(file)].update_tree()
                else:
                    # add
                    self.add_child(TreeNode.build_tree(osp.join(self.path, file)))
    
    def find_path(self, *args: str) -> List[str]:
        """
        Find target path through the provided keys, keys must be sequential

        Args:
            args: provided keys, like "rnnt", "Server"

        Returns:
            return the paths of last keyword if found
        """

        if not self:
            return []

        target = osp.join(*args)
        path = []

        if self.path.endswith(target):
            path.append(self.path)

        if not self.childs:
            return path

        for child in self.childs:
            path += child.find_path(target)

        return path

    def __repr__(self) -> str:
        if self.childs:
            return "\033[1;34m" + self.name + "\033[0m"
        else:
            return "\033[1;37m" + self.name + "\033[0m"

    @classmethod
    def build_tree(cls, path: str) -> "TreeNode":
        """
        Build a tree since path.

        Args:
            path: root path to be build tree

        Returns:
            root node.
        """
        if not osp.exists(path):     # non-exist
            return

        root = TreeNode(path.strip('/').split('/')[-1], path)

        if not osp.isdir(path):
            return root

        for file in os.listdir(path):
            sub_path = osp.join(path, file)

            if osp.isdir(sub_path):
                root.add_child(cls.build_tree(sub_path))
            else:
                root.add_child(TreeNode(file, sub_path))

        return root    

    @classmethod
    def print_tree(cls, root: "TreeNode", sc: int = 0):
        """
        Print tree structure.
        Args:
            root: the root node to be print
            pc: the stack count when call function

        Returns:
            None
        """
        if not root.childs:
            print("{}{}".format("\033[1;37m" + '-'*sc*4 + "\033[0m", root))
            return

        print("{}{}".format("\033[1;37m" + '-'*sc*4 + "\033[0m", root))

        for child in root.childs:
            cls.print_tree(child, sc + 1)
