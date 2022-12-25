from pathlib import Path
from typing import Sequence, Tuple

from manim import (
    BLUE,
    DOWN,
    GOLD,
    GREEN,
    LEFT,
    PURPLE,
    RED,
    RIGHT,
    UP,
    ArrowVectorField,
    Axes,
    DecimalNumber,
    Dot,
    Line,
    NumberPlane,
    Scene,
    Tex,
    ValueTracker,
    VGroup,
)
from numpy import load, ndarray, stack, zeros_like

data_path = Path(__file__).parent.parent / "data" / "ames.npz"

arrays = load(data_path)
xs, ys = arrays["x"], arrays["y"]
coords = stack((xs, ys, zeros_like(xs)), axis=1)

XLIM = (-3, 3)
YLIM = (-3, 3)
SCALE_FORMULAS = 0.7


def mse(theta_0: float, theta_1: float) -> float:
    return ((xs * theta_1 + theta_0 - ys) ** 2).mean()


def predict(theta_0: float, theta_1: float) -> ndarray:
    return theta_0 + xs * theta_1


def residuals(theta_0: float, theta_1: float) -> ndarray:
    return predict(theta_0, theta_1) - ys


def gradient(theta_0: float, theta_1: float) -> Tuple[float, float]:
    r = residuals(theta_0, theta_1)
    g_0 = r.mean()
    g_1 = xs.T.dot(r) / ys.size
    return g_0, g_1


def update(theta_0: float, theta_1: float, lr: float) -> Tuple[float, float]:
    g_0, g_1 = gradient(theta_0, theta_1)
    return theta_0 - lr * g_0, theta_1 - lr * g_1


def get_links(dots: VGroup, predictions: VGroup) -> VGroup:
    return VGroup(*(Line(d, p, color=GREEN) for d, p in zip(dots, predictions)))


class ErrorVectorField(VGroup):
    def __init__(self, theta_0: ValueTracker, theta_1: ValueTracker) -> None:
        super().__init__()
        ax = Axes(
            x_range=(*XLIM, 1),
            y_range=(*YLIM, 1),
            x_length=sum(XLIM),
            y_length=sum(YLIM),
            x_axis_config=dict(color=PURPLE),
            y_axis_config=dict(color=GOLD),
        )
        theta_0_label = Tex(r"$\theta_0$", color=PURPLE).next_to(ax, RIGHT)
        theta_1_label = Tex(r"$\theta_1$", color=GOLD).next_to(ax, UP)
        title = Tex(r"Erreur en fonction des $\theta$").next_to(theta_1_label, UP)
        vector_field = ArrowVectorField(
            func=lambda p: tuple(-x for x in gradient(p[0], p[1])),
            x_range=list(XLIM),
            y_range=list(YLIM),
            opacity=0.5,
        )
        params = Dot(
            ax.c2p(theta_0.get_value(), theta_1.get_value(), 0), color=GREEN, z_index=0
        )
        params.add_updater(lambda p: p.set_x(ax.c2p(theta_0.get_value(), 0, 0)[0]))
        params.add_updater(lambda p: p.set_y(ax.c2p(0, theta_1.get_value(), 0)[1]))
        self.add(
            ax,
            theta_0_label,
            theta_1_label,
            title,
            vector_field,
            params,
        )


class Formulas(VGroup):
    def __init__(self, theta_0: ValueTracker, theta_1: ValueTracker) -> None:
        super().__init__()
        self._theta_0 = theta_0
        self._theta_1 = theta_1

        error_formula = self._create_error_vgroup()
        theta_0_formula = self._create_theta_0_vgroup()
        theta_1_formula = self._create_theta_1_vgroup()
        y_formula = self._create_y_vgroup()

        theta_0_formula.next_to(error_formula, DOWN).align_to(error_formula, LEFT)
        theta_1_formula.next_to(theta_0_formula, DOWN).align_to(error_formula, LEFT)
        y_formula.next_to(theta_1_formula, DOWN).align_to(error_formula, LEFT)

        self.add(error_formula, theta_0_formula, theta_1_formula, y_formula)

    def _create_y_vgroup(self) -> VGroup:
        label = Tex(r"$\hat{y}$ = ", color=RED)

        theta_1 = DecimalNumber(self._theta_1.get_value(), color=GOLD)
        theta_1.add_updater(lambda x: x.set_value(self._theta_1.get_value()))

        between = Tex("$x$", color=RED)
        between.add_updater(lambda x: x.next_to(theta_1, buff=0.1))

        theta_0 = DecimalNumber(
            self._theta_0.get_value(), color=PURPLE, include_sign=True
        )
        theta_0.add_updater(lambda x: x.set_value(self._theta_0.get_value()))
        theta_0.add_updater(lambda x: x.next_to(between, RIGHT))

        theta_1.next_to(label, RIGHT)
        between.next_to(theta_1, RIGHT, buff=0.1)
        theta_0.next_to(between, RIGHT)

        return VGroup(label, theta_1, between, theta_0)

    def _create_error_vgroup(self) -> VGroup:
        label = Tex("E = ", color=GREEN)
        value = DecimalNumber(
            mse(self._theta_0.get_value(), self._theta_1.get_value()), color=GREEN
        )
        value.add_updater(
            lambda x: x.set_value(
                mse(self._theta_0.get_value(), self._theta_1.get_value())
            )
        )
        value.next_to(label, RIGHT)
        return VGroup(label, value)

    def _create_theta_vgroup(
        self, tracker: ValueTracker, label: str, color: str
    ) -> VGroup:
        label = Tex(label, color=color)
        value = DecimalNumber(tracker.get_value(), color=color)
        value.add_updater(lambda x: x.set_value(tracker.get_value()))
        value.next_to(label, RIGHT)
        return VGroup(label, value)

    def _create_theta_0_vgroup(self) -> VGroup:
        return self._create_theta_vgroup(self._theta_0, r"$\theta_0$ = ", PURPLE)

    def _create_theta_1_vgroup(self) -> VGroup:
        return self._create_theta_vgroup(self._theta_1, r"$\theta_1$ = ", GOLD)


class Data(VGroup):
    def __init__(self, theta_0: ValueTracker, theta_1: ValueTracker) -> None:
        super().__init__()
        self._theta_0 = theta_0
        self._theta_1 = theta_1
        number_plane = NumberPlane(x_range=(*XLIM, 1), y_range=(*YLIM, 1))
        dots = VGroup(*(Dot(c, color=BLUE) for c in coords))
        number_plane.add(dots)
        title = Tex("Données").next_to(number_plane, UP).shift(UP * 0.75)
        line = Line(color=RED).add_updater(
            lambda line: line.put_start_and_end_on(
                number_plane.c2p(XLIM[0], self._predict(XLIM[0]), 0),
                number_plane.c2p(XLIM[1], self._predict(XLIM[1]), 0),
            )
        )
        preds = VGroup(
            *(
                Dot(
                    number_plane.c2p(c[0], self._predict(c[1]), 0), color=RED
                ).add_updater(
                    lambda d: d.set_y(
                        number_plane.c2p(
                            0, self._predict(number_plane.p2c(d.get_center())[0]), 0
                        )[1]
                    )
                )
                for c in coords
            )
        )

        def _create_line(dot: Dot, pred: Dot) -> Line:
            return Line(dot, pred, color=GREEN).add_updater(
                lambda line: line.put_start_and_end_on(
                    dot.get_center(),
                    number_plane.c2p(
                        number_plane.p2c(dot.get_center())[0],
                        self._predict(number_plane.p2c(dot.get_center())[0]),
                    ),
                )
            )

        errors = VGroup(*(_create_line(dot, pred) for dot, pred in zip(dots, preds)))
        self.add(number_plane, title, line, preds, errors)

    def _predict(self, x: float) -> float:
        return self._theta_0.get_value() + self._theta_1.get_value() * x


class ErrorSurface(Scene):
    def construct(self) -> None:

        theta_0, theta_1 = ValueTracker(0.0), ValueTracker(0.0)

        error_vgroup = ErrorVectorField(theta_0, theta_1)
        error_vgroup.shift(RIGHT * 4).scale(SCALE_FORMULAS)

        formulas_vgroup = Formulas(theta_0, theta_1)
        formulas_vgroup.shift(UP + LEFT * 1.8).scale(SCALE_FORMULAS)

        data_vgroup = Data(theta_0, theta_1)
        data_vgroup.shift(LEFT * 4.5).scale(SCALE_FORMULAS)

        self.add(data_vgroup, formulas_vgroup, error_vgroup)

        self.play(theta_0.animate.set_value(2))
        self.play(theta_0.animate.set_value(-2))
        self.play(theta_0.animate.set_value(0))

        self.wait(1)
        self.play(theta_1.animate.set_value(2))
        self.play(theta_1.animate.set_value(-2))
        self.play(theta_1.animate.set_value(0))

        self.wait(1)
        self.play(theta_0.animate.set_value(-0.98), theta_1.animate.set_value(-0.76))

        for _ in range(15):
            self.wait(1)
            new_theta_0, new_theta_1 = update(
                theta_0.get_value(), theta_1.get_value(), 0.2
            )
            self.play(
                theta_0.animate.set_value(new_theta_0),
                theta_1.animate.set_value(new_theta_1),
            )
        self.wait(4)
