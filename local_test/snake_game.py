from dataclasses import dataclass
from enum import Enum
from random import sample
import random
import numpy as np
import cv2

class SnakeState(Enum):
    OK = 1
    ATE = 2
    DED = 3
    WON = 4

class Colour(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
    YELLOW = 4

    def get_RGB(self, gradation=1):
        if self == Colour.BLUE:
            return (255*gradation, 0*gradation, 0*gradation)
        if self == Colour.GREEN:
            return (0*gradation, 255*gradation, 0*gradation)
        if self == Colour.RED:
            return (0*gradation, 0*gradation, 255*gradation)

@dataclass(eq=True, frozen=True)
class Point:
    x: int
    y: int

    def copy(self, xincr, yincr):
        return Point(self.x + xincr, self.y + yincr)

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d['x'], d['y'])

    def __repr__(self):
        return f"(x: {self.x}, y: {self.y})"

    def __sub__(self, other):
        return Point(self.x-other.x, self.y-other.y)

    def dist(self, other):
        m = Point(self.x-other.x, self.y-other.y)
        return abs(m.x) + abs(m.y)

class Direction(Enum):
    UP = Point(0, -1)
    DOWN = Point(0, 1)
    LEFT = Point(-1, 0)
    RIGHT = Point(1, 0)

    def to_int(self):
        match self:
            case Direction.UP:
                return 1
            case Direction.DOWN:
                return 2
            case Direction.LEFT:
                return 3
            case Direction.RIGHT:
                return 4

    def x(self):
        return self.value.x
    
    def y(self):
        return self.value.y

    def turn_left(self):
        if self == Direction.UP:
            return Direction.LEFT
        if self == Direction.LEFT:
            return Direction.DOWN
        if self == Direction.DOWN:
            return Direction.RIGHT
        if self == Direction.RIGHT:
            return Direction.UP
    
    def turn_right(self):
        if self == Direction.UP:
            return Direction.RIGHT
        if self == Direction.RIGHT:
            return Direction.DOWN
        if self == Direction.DOWN:
            return Direction.LEFT
        if self == Direction.LEFT:
            return Direction.UP

dir_map_to_angle = {
    Direction.UP: 0,
    Direction.DOWN: 180,
    Direction.LEFT: 90,
    Direction.RIGHT: -90,
}

sprites = {
    'head': cv2.imread('./sprites/head.png', 1),
    'body': cv2.imread('./sprites/body.png', 1),
    'fruit': cv2.imread('./sprites/fruit.png', 1),
    'tail': cv2.imread('./sprites/tail.png', 1),
}

def _rotate_image(cv_image, _rotation_angle):
    axes_order = (1, 0, 2) if len(cv_image.shape) == 3 else (1, 0)
    if _rotation_angle == -90:
        return np.transpose(cv_image, axes_order)[:, ::-1]

    if _rotation_angle == 90:
        return np.transpose(cv_image, axes_order)[::-1, :]

    if _rotation_angle in [-180, 180]:
        return cv_image[::-1, ::-1]

    return cv_image


INIT_TAIL_SIZE = 2
class Snake:
    def __init__(self, x: int = 0, y: int = 0, health=10, colour=Colour.RED):
        self.head = Point(x, y)
        self.tail = []
        self.tail_size = INIT_TAIL_SIZE
        self.direction = Direction.UP  # Need to add validation later
        self.dir_idx = 0
        self.hp = health
        self.colour = colour 

    def self_collision(self):
        for t in self.tail:
            if self.head.x == t.x and self.head.y == t.y:
                return True
        return False

    # def to_dict(self):
    #     return {
    #         'head': self.head.to_dict(),
    #         'tail': [t.to_dict() for t in self.tail],
    #         'tail_size': self.tail_size,
    #         'direction': self.direction.to_dict()
    #     }

    # @classmethod
    # def from_dict(cls, d):
    #     s = cls()
    #     s.head = Point.from_dict(d['head'])
    #     s.tail = [Point.from_dict(t) for t in d['tail']]
    #     s.tail_size = d['tail_size']
    #     s.direction = Point.from_dict(d['direction'])
    #     return s

    def update(self, decay=False):
        new_head = self.head.copy(self.direction.x(), self.direction.y())

        self.tail.append(self.head)
        self.head = new_head

        if decay:
            self.hp -= 1

    def shed(self):
        if self.tail_size > 0:
            self.tail = self.tail[-self.tail_size:]
        else:
            self.tail = []

    def __repr__(self):
        return f"""Head: {self.head}
        Tail: {self.tail}
        Dir: {self.direction}
        """

    def apply_direction(self, action=['left', 'right', 'stay']):
        # if action == 'stay':
        #     return
        if action == 'left':
            self.direction = self.direction.turn_left()
        elif action == 'right':
            self.direction = self.direction.turn_right()

class Env:
    def __init__(self, grid_size=10, num_fruits=10, num_snakes=1, num_teams=1, init_hp=10):
        self.gs = grid_size
        self.num_fruits = num_fruits
        self.num_snakes = num_snakes
        self.num_teams = num_teams
        self.time_steps = 0
        self.decay_rate = 1
        self.init_hp = init_hp

        self.reset()

    def reset(self):
        self.step = 0
        grid_size = self.gs

        self.snakes = [Snake(random.randint(0, self.gs-1), random.randint(0, self.gs-1), health=self.init_hp) for _ in range(self.num_snakes)]
        if self.num_teams == 2:
            #TODO: Implement team logic
            for i, snake in enumerate(self.snakes):
                if (i+1) % 2 == 0:
                    snake.colour = Colour.BLUE

        pos_list = []
        for i in range(grid_size):
            for j in range(grid_size):
                pos_list.append(Point(i, j))

        self.pos_set = set(pos_list)
        self.fruit_locations = []
        self.set_fruits()

        self.time_steps = 0


    # def to_dict(self):
    #     return {
    #         'snake': self.snake.to_dict(),
    #         'fruit': self.fruit_loc.to_dict()
    #     }


    # def from_dict(self, d):
    #     self.snake = Snake.from_dict(d['snake'])
    #     self.fruit_location = Point.from_dict(d['fruit'])

    def get_snake_locs(self):
        snake_locs = []
        for snake in self.snakes:
            snake_locs.append(snake.head)
            snake_locs.extend(snake.tail)
        return snake_locs

    def update(self, directions:list[str]):
        hp_decay = self.decay_rate and self.time_steps % self.decay_rate == 0

        snake_states = []
        for snake, direction in zip(self.snakes, directions):
            snake.apply_direction(direction)
            snake.update(hp_decay)
            snake_condition = SnakeState.OK

            if snake.head in self.fruit_locations:
                self.fruit_locations.pop(self.fruit_locations.index(snake.head))
                snake.tail_size += 1
                snake.hp += 10
                snake_condition = SnakeState.ATE
            
            snake.shed()
            if not self._bounds_check(snake.head) or snake.self_collision() or snake.hp <= 0:
                snake_condition = SnakeState.DED

            snake_states.append(snake_condition)

        self.snake_locs = self.get_snake_locs()

        # check collision with other snakes
        if self.num_snakes > 1:
            for i, snake in enumerate(self.snakes):
                locs = self.snake_locs.copy()
                locs.remove(snake.head)
                if snake.head in locs:
                    snake_states[i] = SnakeState.DED
        
        dead_snakes = [i+1 for i, state in enumerate(snake_states[1:]) if state == SnakeState.DED]
        for dead_snake in dead_snakes:
            self.snakes.pop(dead_snake)
        
        self.set_fruits()
        self.time_steps += 1
        return snake_states[0], self.snakes[0].hp, self.snakes[0].tail_size

    @property
    def fruit_loc(self):
        return self.fruit_locations

    def set_fruits(self):
        snake_locs = self.get_snake_locs()
        snake_locs.extend(self.fruit_locations)
        possible_positions = [pos for pos in self.pos_set if pos not in snake_locs]
        diff = self.num_fruits - len(self.fruit_locations)
        new_locs = sample(list(possible_positions), k=min(diff, len(possible_positions)))
        self.fruit_locations.extend(new_locs)

    def _bounds_check(self, pos):
        return pos.x >= 0 and pos.x < self.gs and pos.y >= 0 and pos.y < self.gs

    def to_image(self, gradation=True):
        fl = self.fruit_loc
        scale = 8

        canvas = np.zeros((self.gs*scale, self.gs*scale, 3), 'uint8')
        h, w = self.gs*scale, self.gs*scale

        def apply_rotation(im, angle):
            return _rotate_image(im, angle)

        def draw_body(canvas, y, x, colour, alpha=1):
            s = scale
            canvas[y*s:(y+1)*s, x*s:(x+1)*s] = colour.get_RGB(alpha)

        def draw_sprite(canvas, y, x, stype, scale=8, rotation=0):
            s = scale
            canvas[y*s:(y+1)*s, x*s:(x+1)*s] = apply_rotation(sprites[stype], rotation)

        for f in fl:
            draw_sprite(canvas, f.y, f.x, 'fruit')

        for snake in self.snakes:
            if self._bounds_check(snake.head):
                draw_sprite(canvas, snake.head.y, snake.head.x, 'head',
                            rotation=dir_map_to_angle[snake.direction])

            limbs = list(reversed(snake.tail))
            if gradation:
                for i, limb in enumerate(limbs):
                    draw_body(canvas, limb.y, limb.x, snake.colour, 1-0.5*(i/len(limbs)))
            else:
                for limb in limbs:
                    draw_body(canvas, limb.y, limb.x, snake.colour)

        return canvas
    
if __name__ == '__main__':
    env = Env(10, 2, 2, 1)
    cv2.imwrite('test/test.png', cv2.resize(env.to_image(), (640, 640), interpolation=cv2.INTER_NEAREST)) 

    while True:
        n = input()
        print(env.update([n, 'stay']))
        cv2.imwrite('test/test.png', cv2.resize(env.to_image(), (640, 640), interpolation=cv2.INTER_NEAREST))