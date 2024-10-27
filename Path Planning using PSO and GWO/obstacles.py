import math
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

#class to make circular obstacles
class Obstacle_Circle:

    def __init__(self, r, C):
        self.r = r #radius
        self.C = C #centre

    def inside_circle(self, p):

        return (p.x - self.C.x) ** 2 + (p.y - self.C.y) ** 2 <= self.r**2 + 1

    def how_to_exit_y(self, y):

        return y - self.C.y

    def how_to_exit_x(self, x):

        return x - self.C.x