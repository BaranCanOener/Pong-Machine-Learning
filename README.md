# Pong-Machine-Learning
Pong game &amp; supervised learning-based AI to predict ball impact coordinates.

![Pong](https://github.com/BaranCanOener/Pong-Machine-Learning/blob/main/screenshot.png)

A Python-implementation of Pong, allowing for player-vs-player and player-vs-computer, where the computer AI that controls the paddle either does so in a hardcoded way, or via a supervised machine learning model that was implemented using Keras.

- The hardcoded AI calculates the trajectory of the ball to predict the exact point of impact and moves the paddle to that point (plus/minus a random deviation to make for more interesting play).
- The machine learning model is designed to predict the exact point of impact, by taking as an impact vector the 2d velocity vector of the ball as well as its vertical position on the game field. The player can cumulatively train this model on 1000 balls by pressing 't'.

Hence, the machine learning model boils down to is an approximation of the function R^3 ---> R that maps from the ball phase space to the impact y-coordinate space, governed by "Pong physics": the simple perfectly elastic no-drag-no-gravity collision model, where impact distance to the paddle's middle dictates the angle at which the ball bounces back (the further away from the middle, the more acute the angle; the closer to the middle, the more obtuse it is).
The input data is simplified by assuming a normalized velocity vector, as well as taking (x,y) positions only at the start of the field - and the training data is generated accordingly. An interesting exercise could be how relaxing this assumption impacts the needed training data size/model complexity to make it perform well.

As my first machine learning project, my main takeaways were:
- Why do this in the first place? It's a little case study to explore how neural nets can approximate arbitrary mathematical functions, but (a) that's well known and grounded in theory and (b) this particular problem is so perfectly amenable to an explicit model, that an elaborate approximation of the same seems somewhat pointless.
In fact, the "ray-tracing"-approach to predicting the ball trajectory is already more convoluted than it needs to be; under the simple Pong physics, solving a set of linear equations is enough to exactly determine the ball's point of impact.
- Properly scaled input data is important, as is data cleaning as a whole. I initially did not normalize the velocity vector, yielding much worse results - let alone storing the positional data in terms of pixels rather than normalized world coordinates (0..1 for the verical axis).
- It is very easy to make the hardcoded AI play interestingly - for example, by making it jittery and unpredictable using random quantities; one could also go as far as to employ different strategies, e.g. aiming always for the paddle edges to achieve a very acute angle, or hitting the ball in a way that maximizes the predicted impact's distance to the opponent's paddle (which is what a tennis player in real life might do. this, though, is nontrivial here due to the discrete intervals at which the paddle moves, where a minimal massively impacts the ball's trajectory, making it difficult to aim precisely).
Making the Machine Learning model develop a workable strategy on its own, however, would require a different approach, potentially using unsupervised/reinforcement learning.

Things to implement or fix:
- Sometimes the model still predicts inaccurately, even after training it on 50.000+ data points - it occasionally seems to undershoot at very obtuse angles. The training data may be improved, or perhaps the model's hyperparameters warrant some tuning.
- The Pong physics could be improved by detecting when the ball hits the short edge of the paddle. The game in its current state is very forgiving; it is a very simple implementation of the collision physics. The dropping balls repo could serve as a better basis.
- Visuals etc.
