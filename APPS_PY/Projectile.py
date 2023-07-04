# Packages
import numpy as np
from shiny import App, ui, run_app, render, Inputs, Outputs, Session, reactive
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import asyncio

# UI
app_ui = ui.page_fluid(

    # add head that allows LaTeX to be displayed via MathJax
    ui.head_content(
        ui.tags.script(
            src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
        ),
        ui.tags.script(
            "if (window.MathJax) MathJax.Hub.Queue(['Typeset', MathJax.Hub]);"
        ),
    ),

    # Panel on top for title
    ui.panel_title(
        "Projectile simulation"
    ),
    ui.p("""In classical physics, a projectile is an object that is thrown or launched into the air and moves along
         a curved path under the influence of gravity. In this simulation, we will consider
         a projectile motion without air resistance."""),

    ui.p("The vertical position of the projectile at time \\( t \\) can be calculated using the following formula:"),
    ui.p("$$ y(t) = y_0 + v_0 \\cdot \\sin(\\theta) \\cdot t - \\frac{1}{2} \\cdot g \\cdot t^2 $$"),
    ui.tags.ul(
        ui.tags.li("where \\( y(t) \\) is the vertical position at time \\( t \\)"),
        ui.tags.li("where \\( y_0 \\) is the initial vertical position (height from the ground)"),
        ui.tags.li("where \\( v_0 \\) is the initial velocity magnitude"),
        ui.tags.li("where \\( \\theta \\) is the launch angle"),
        ui.tags.li("where \\( t \\) is the time"),
        ui.tags.li("where \\( g \\) is the acceleration due to gravity (approximately 9.8 m/s^2)")
    ),

    ui.p("The horizontal position of the projectile at time \\( t \\) can be calculated as:"),
    ui.p("$$ x(t) = x_0 + v_0 \\cdot \\cos(\\theta) \\cdot t $$"),
    ui.tags.ul(
        ui.tags.li("where \\( x(t) \\) is the horizontal position at time \\( t \\)"),
        ui.tags.li("where \\( x_0 \\) is the initial horizontal position (usually considered as 0)"),
        ui.tags.li("where \\( \\theta \\) is the launch angle"),
        ui.tags.li("where \\( t \\) is the time")

    ),

    ui.p("""In this simulation, you will be able to adjust the gravitational constant, the launch angle, 
    and the initial velocity to observe the resulting trajectory of the projectile. 
    Keep in mind that we are neglecting air resistance and assuming no acceleration 
    or deceleration during the flight.
    Enjoy exploring the projectile motion simulation!"""),

    # Defining sidebars
    ui.layout_sidebar(

        # Panel on the let side for the inputs
        ui.panel_sidebar(

            # Initial conditions
            ui.input_numeric(id="xstart", value=0, label="\\(x_0\\) value (m)"),
            ui.input_numeric(id="ystart", value=100, label="\\(y_0\\) value (m)"),
            ui.input_numeric(id="vstart", value=30, label="\\(v_0\\) (initial velocity) value in m/s",
                             min=0, max=100),
            ui.input_numeric(id="thetastart", value=45, label="\\( \\theta \\) value in degrees",
                             min=0, max=360),
            ui.input_numeric(id="g", value=9.80665, label="Gravitational constant \\( m/s^2 \\)",
                             min=0, max=360),
            ui.p(ui.input_action_button("simul", "Start simulation"))
        ),

        # Panel on the right for the simulation and explanation
        ui.panel_main(
            ui.h1("Simulation !"),
            ui.p("""
                The simulation will pop in another window when you press the button to start !
                
                Enjoy!
            """),
            ui.output_text("text")
        ),
    ),
)


# Server
def server(input, output, session):
    @reactive.Calc
    # Redefine the inputs
    def x0():
        return input.xstart()

    def y0():
        return input.ystart()

    def v0():
        return input.vstart()

    def g():
        return input.g()

    def angle():
        return np.pi * input.thetastart() / 180

    print("Arg done")

    # Moment of impact ! We are solving the quadratic
    def calculate_last_time():

        # Check if the initial vertical position is above the ground (ystart > 0)
        discriminant = (v0() * np.sin(angle())) ** 2 - 4 * (-0.5 * g()) * y0()

        # Check if the discriminant is positive to have real solutions
        if discriminant >= 0:
            t1 = (-v0() * np.sin(angle()) + np.sqrt(discriminant)) / (2 * (-0.5 * g()))
            t2 = (-v0() * np.sin(angle()) - np.sqrt(discriminant)) / (2 * (-0.5 * g()))

            # Take the positive time value
            last_time = max(t1, t2, 0)
            return float(last_time)
        else:
            return 0  # Return 0 if there are no real solutions

    def final_time():
        return calculate_last_time()

    # All the times evaluated | keeping approx 60 fps
    def tt():
        return np.linspace(start=0, stop=final_time(), num=int(60 * final_time()))

    # The y(t) function
    def y(t):
        return y0() + np.sin(angle()) * v0() * t - g()/2 * t * t

    # The x(t) function
    def x(t):
        return v0() * np.cos(angle()) * t + x0()

    # Calculate all the points
    def xx():
        return x(tt())

    def yy():
        return y(tt())

    print("Anim prep done")

    # Animation
    def animate(i):
        plt.cla()  # Clear the current plot

        # Plot the projectile trajectory
        plt.plot(xx()[i], yy()[i], color='blue', marker='o')

        # Add the floor at y = 0
        plt.axhline(0, color='gray', linestyle='--')

        # Add a tracer that follows the previous positions
        if i > 0:
            plt.plot(xx()[:i+1], yy()[:i+1], color='red')

        # Set the plot limits
        plt.xlim(np.min(xx()), np.max(xx()) + 10)
        plt.ylim(-10, np.max(yy()) + 10)

        # Set the axis labels
        plt.xlabel('Horizontal position')
        plt.ylabel('Vertical position')

        # Set the title
        plt.title('Projectile Motion Simulation')

    @output
    @render.text()
    @reactive.event(input.simul)
    def text():
        fig = plt.figure()
        anim = FuncAnimation(fig, animate, frames=len(tt()),
                             interval=final_time() / (len(tt()) * 1000))
        plt.show()  # Display the animation in a separate window
        return "Animation ready :)"


# App object
app = App(app_ui, server)

# Run the app
run_app(app)
