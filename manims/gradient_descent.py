from pathlib import Path
from typing import Tuple

from manim import (
    BLUE,
    DEGREES,
    DOWN,
    GOLD,
    GREEN,
    GREEN_D,
    GREEN_E,
    LEFT,
    PURPLE,
    RED,
    RIGHT,
    UP,
    DashedLine,
    Dot,
    Line,
    MathTex,
    NumberPlane,
    Surface,
    Tex,
    ThreeDAxes,
    ThreeDScene,
    Transform,
    VGroup,
)
from numpy import array, load, ndarray, stack, zeros_like

data_path = Path(__file__).parent.parent / "data" / "ames.npz"

arrays = load(data_path)
xs, ys = arrays["x"], arrays["y"]
coords = stack((xs, ys, zeros_like(xs)), axis=1)

XLIM_2D = (-2.5, 2.5)
YLIM_2D = (-2.5, 2.5)

XLIM_3D = (-3, 3)
YLIM_3D = (-3, 3)
SHIFT_LEFT = LEFT * 4.5 + UP * 1.5
ALIGN_LEFT = LEFT * 7
SHIFT_RIGHT = RIGHT * 6 + DOWN * 3


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


def init() -> Tuple[float, float]:
    return -0.98, -0.76


def get_line(theta_0: float, theta_1: float) -> Line:
    return Line(
        array([XLIM_2D[0], theta_0 + theta_1 * XLIM_2D[0], 0]) + SHIFT_LEFT,
        array([XLIM_2D[1], theta_0 + theta_1 * XLIM_2D[1], 0]) + SHIFT_LEFT,
        color=RED,
    )


def get_predictions(theta_0: float, theta_1: float) -> VGroup:
    return VGroup(
        *(
            Dot(array([x, theta_0 + theta_1 * x, 0]) + SHIFT_LEFT, color=RED, z_index=1)
            for x in xs
        )
    )


def get_links(dots: VGroup, predictions: VGroup) -> VGroup:
    return VGroup(*(Line(d, p, color=GREEN) for d, p in zip(dots, predictions)))


def get_y_formula(theta_0: float, theta_1: float) -> MathTex:
    theta_1_tex = f"{theta_1:.2f}"
    theta_0_tex = f"- {-theta_0:.2f}" if theta_0 < 0 else f"+ {theta_0:.2f}"
    tex = (
        MathTex(r"\hat{y} = ", theta_1_tex, " x ", theta_0_tex, color=RED)
        .scale(0.75)
        .align_to(ALIGN_LEFT, LEFT)
        .shift(DOWN * 3)
    )
    tex[1].set_color(GOLD)
    tex[3].set_color(PURPLE)
    return tex


def get_loss_formula(loss: float) -> Tex:
    return (
        Tex(rf"$E = {loss:.2f}$", color=GREEN)
        .scale(0.75)
        .align_to(ALIGN_LEFT, LEFT)
        .shift(DOWN * 1.8)
    )


def get_theta_0_formula(theta_0: float) -> Tex:
    return (
        Tex(f"$b = {theta_0:.2f}$", color=PURPLE)
        .scale(0.75)
        .align_to(ALIGN_LEFT, LEFT)
        .shift(DOWN * 2.2)
    )


def get_theta_1_formula(theta_1: float) -> Tex:
    return (
        Tex(f"$a = {theta_1:.2f}$", color=GOLD)
        .scale(0.75)
        .align_to(ALIGN_LEFT, LEFT)
        .shift(DOWN * 2.6)
    )


class ErrorSurface(ThreeDScene):
    def construct(self) -> None:
        axes = ThreeDAxes(
            x_range=(*XLIM_3D, 1),
            y_range=(*YLIM_3D, 1),
            z_range=(0, 6, 1),
            x_axis_config=dict(color=PURPLE),
            y_axis_config=dict(color=GOLD),
            z_axis_config=dict(color=GREEN),
        )
        axes.shift(SHIFT_RIGHT)
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        error_surface = Surface(
            lambda u, v: array(
                [
                    u,
                    v,
                    mse(u, v),
                ]
            ),
            u_range=XLIM_3D,
            v_range=YLIM_3D,
            checkerboard_colors=(GREEN_D, GREEN_E),
        )
        error_surface.set_opacity(0.3)
        error_surface.shift(SHIFT_RIGHT)
        self.add(axes, error_surface)

        number_plane = NumberPlane(x_range=(*XLIM_2D, 1), y_range=(*YLIM_2D, 1))
        self.dots = VGroup(*(Dot(c, color=BLUE) for c in coords))
        number_plane.add(self.dots)
        number_plane.shift(SHIFT_LEFT)
        self.add_fixed_in_frame_mobjects(number_plane)

        theta_0, theta_1 = 0.0, 0.0
        loss = mse(theta_0, theta_1)
        self.y_formula = get_y_formula(theta_0, theta_1)
        self.theta_0_formula = get_theta_0_formula(theta_0)
        self.theta_1_formula = get_theta_1_formula(theta_1)
        self.loss_formula = get_loss_formula(loss)
        self.params_dot = Dot([theta_0, theta_1, 0], color=RED)
        self.loss_dot = Dot([theta_0, theta_1, loss], color=RED)
        self.loss_line = DashedLine(
            self.params_dot,
            self.loss_dot,
            color=RED,
        )
        self.loss_dot.shift(SHIFT_RIGHT)
        self.params_dot.shift(SHIFT_RIGHT)
        self.loss_line.shift(SHIFT_RIGHT)
        self.line = get_line(theta_0, theta_1)
        self.predictions = get_predictions(theta_0, theta_1)
        self.links = get_links(self.dots, self.predictions)
        self.add(self.loss_dot, self.loss_line, self.params_dot)
        self.add_fixed_in_frame_mobjects(
            self.line,
            self.predictions,
            self.links,
            self.y_formula,
            self.theta_0_formula,
            self.theta_1_formula,
            self.loss_formula,
        )

        self._update_from_thetas(2, theta_1)
        self._update_from_thetas(-2, theta_1)
        self._update_from_thetas(theta_0, theta_1)
        self._update_from_thetas(theta_0, 2)
        self._update_from_thetas(theta_0, -2)
        self._update_from_thetas(theta_0, theta_1)

        theta_0, theta_1 = init()
        self._update_from_thetas(theta_0, theta_1)
        self.wait(1)
        for n in range(15):
            self.wait(1)
            theta_0, theta_1 = update(theta_0, theta_1, 0.2)
            self._update_from_thetas(theta_0, theta_1)
        self.wait(4)

    def _update_from_thetas(self, theta_0: float, theta_1: float) -> None:
        loss = mse(theta_0, theta_1)
        new_loss_dot = Dot(
            array([theta_0, theta_1, loss]) + SHIFT_RIGHT,
            color=RED,
        )
        new_params_dot = Dot(array([theta_0, theta_1, 0]) + SHIFT_RIGHT, color=RED)
        new_predictions = get_predictions(theta_0, theta_1)
        self.play(
            Transform(self.params_dot, new_params_dot),
            Transform(self.loss_dot, new_loss_dot),
            Transform(
                self.loss_line,
                DashedLine(new_params_dot, new_loss_dot, color=RED),
            ),
            Transform(self.line, get_line(theta_0, theta_1)),
            Transform(self.predictions, new_predictions),
            Transform(self.links, get_links(self.dots, new_predictions)),
            Transform(self.y_formula, get_y_formula(theta_0, theta_1)),
            Transform(self.theta_0_formula, get_theta_0_formula(theta_0)),
            Transform(self.theta_1_formula, get_theta_1_formula(theta_1)),
            Transform(self.loss_formula, get_loss_formula(loss)),
        )
