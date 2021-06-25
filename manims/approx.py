from functools import partial
from typing import Callable

from manim.constants import DOWN, LEFT, RIGHT
from manim.utils.color import BLUE, GREEN, RED, WHITE
from manim import (
    ApplyMethod,
    Axes,
    Circle,
    Create,
    FadeOut,
    Line,
    MathTex,
    Scene,
    Tex,
    Transform,
    Underline,
    VGroup,
    Write,
)
from numpy import arange


class UniversalApproximation(Scene):
    def construct(self) -> None:
        axes = Axes(
            x_range=[-2, 2.01, 0.5],
            y_range=[-2, 2.01, 0.5],
            x_length=8,
            y_length=6,
            axis_config=dict(
                include_ticks=True,
                numbers_to_include=arange(-2, 2.01),
                numbers_with_elongated_ticks=arange(-2, 2.01),
                decimal_number_config=dict(num_decimal_places=1),
                color=GREEN,
            ),
            tips=False,
        ).shift(2 * LEFT)
        axes.num_sampled_graph_points_per_tick = 100
        cubic_graph = axes.get_graph(lambda x: x ** 3 + x ** 2 - x - 1, color=RED)
        cubic_label = (
            Tex("$f = x^3 + x^2 - x - 1$", color=RED)
            .scale(0.7)
            .align_to(axes, RIGHT)
            .shift(DOWN * 3)
        )
        self.add(axes)
        self.play(Create(cubic_graph), Write(cubic_label))
        self.wait(1)

        def approx(x: float) -> float:
            return 0

        neurons = (
            (-5, -7.7),
            (-1.2, -1.3),
            (1.2, 1),
            (1.2, -0.2),
            (2, -1.1),
            (5, -5),
        )

        coefs = (-1, -1, -1, 1, 1, 1)

        relus = tuple(
            partial(lambda x, w, b: max(0, w * x + b), w=w, b=b) for w, b in neurons
        )

        circle = Circle(radius=0.25, color=BLUE)
        text = Tex("$x$").align_to(circle).scale(0.75)
        input_neuron = VGroup(circle, text)
        input_neuron.next_to(axes)

        hidden_layer = VGroup()
        for i in range(1, 7):
            circle = Circle(radius=0.25, color=BLUE)
            text = Tex(f"$n_{i}$").align_to(circle).scale(0.75)
            hidden_layer += VGroup(circle, text).set_opacity(0.5)

        hidden_layer.arrange_in_grid(6, 1, buff=0.5)
        hidden_layer.next_to(input_neuron, RIGHT).shift(RIGHT)

        circle = Circle(radius=0.25, color=BLUE)
        text = Tex(r"$\hat{f}$").align_to(circle).scale(0.75)
        output_neuron = VGroup(circle, text)
        output_neuron.next_to(hidden_layer, RIGHT).shift(RIGHT)

        nn = VGroup(input_neuron, hidden_layer, output_neuron)

        approx_graph = axes.get_graph(approx, color=BLUE)
        approx_label = (
            MathTex(
                r"\hat{f} = ",
                "-",
                "n_1 ",
                "-",
                "n_2",
                "-",
                "n_3",
                "+",
                "n_4",
                "+",
                "n_5",
                "+",
                "n_6",
                color=BLUE,
            )
            .scale(0.7)
            .align_to(axes, RIGHT)
            .shift(DOWN * 3.5)
        )
        self.play(Create(approx_graph), Write(approx_label), Create(nn))

        for i, (relu, w, neuron) in enumerate(zip(relus, coefs, hidden_layer), start=1):
            relu_graph = axes.get_graph(
                relu, use_smoothing=False, num_sampled_graph_points_per_tick=10000
            )
            start, end = input_neuron.get_boundary_point(
                RIGHT
            ), neuron.get_boundary_point(LEFT)
            line = Line(start, end)
            underline = Underline(approx_label[i * 2], color=WHITE)
            self.play(Create(relu_graph), Create(line), Create(underline))
            self.play(ApplyMethod(neuron.set_opacity, 1))

            if w != 1:
                self.play(
                    Transform(relu_graph, axes.get_graph(lambda x: w * relu(x))),
                    Transform(
                        underline,
                        Underline(approx_label[i * 2 - 1 : i * 2 + 1], color=WHITE),
                    ),
                )

            approx = self.update_approx(approx, relu, w)  # type: ignore
            start, end = neuron.get_boundary_point(
                RIGHT
            ), output_neuron.get_boundary_point(LEFT)
            line = Line(start, end)
            self.play(
                Transform(
                    approx_graph,
                    axes.get_graph(approx, color=BLUE),
                ),
                FadeOut(relu_graph),
                Create(line),
                Transform(
                    underline,
                    Underline(approx_label[: i * 2 + 1], color=BLUE),
                ),
            )
            self.play(FadeOut(underline))

    @staticmethod
    def update_approx(
        approx: Callable[[float], float], relu: Callable[[float], float], w: float
    ) -> Callable[[float], float]:
        def new_approx(x: float) -> float:
            return approx(x) + w * relu(x)

        return new_approx
