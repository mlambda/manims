dev-gradient-descent:
	manim -pql manims/gradient_descent.py ErrorSurface

gradient-descent:
	manim -pqh manims/gradient_descent.py ErrorSurface

.PHONY: dev-gradient-descent
