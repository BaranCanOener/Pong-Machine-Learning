# -*- coding: utf-8 -*-
"""
@author: baran
"""


import os
import random, math
import numpy as np
import tensorflow.config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

white = (255, 255, 255)
red = (255, 0, 0)
darkred = (127, 0, 0)
blue = (0, 0, 255)
black = (0, 0, 0)

class GameState:
    RUNNING = 0
    P1WON = 1
    P2WON = 2
    
class PlayerType:
    HUMAN = 0
    ML_AI = 1
    HARDCODED_AI = 2

"""
Main game class.
- Encapsulates the game logic:
    storing of world coordinates, updating the ball position and velocity per frame,
    controlling the game state (i.e. saving whether the game is ongoing or whether a player has won),
    allowing for paddle controls up/down for each player.
- Stores the position and velocity of the ball when it hit the paddle, which is used by the
  supervised learning algo to predict the point of impact
- Contains an algorithm to predict the trajectory of the ball until point of impact at the targeted players field,
  allowing to (a) have a hardcoded AI and (b) generate training data for the Machine Learning AI
"""
class Pong:
    
    pixels_per_unit = 400
    
    worldspace_left = 0
    worldspace_right = 1.25
    worldspace_top = 0
    worldspace_bottom = 1
    
    worldspace_width = worldspace_right - worldspace_left
    worldspace_height = worldspace_bottom - worldspace_top
    
    paddle_width = 0.05
    paddle_height = 0.2
    
    paddle_posX_p1 = worldspace_left
    paddle_posY_p1 = worldspace_top + worldspace_height / 2 - paddle_height / 2
    
    paddle_posX_p2 = worldspace_right - paddle_width
    paddle_posY_p2 = worldspace_top + worldspace_height / 2 - paddle_height / 2
    
    #How many worldspace units the paddle moves per millisecond
    paddle_unitsPerTick = 0.0015
    
    ball_posX = worldspace_left + paddle_width
    ball_posY = worldspace_top + worldspace_height / 2
    ball_radius = 0.025
    ball_vX = +0.0005
    ball_vY = 0.0
    #How many worldspace units the ball moves per millisecond (increased over time)
    ball_unitsPerTick_default = 0.001
    ball_unitsPerTick = ball_unitsPerTick_default
    
    
    #Player 1 has her paddle on the left side of the field, player 2 is to the right
    score_p1 = 0
    score_p2 = 0
    
    state = GameState.RUNNING
    
    #Stores the vertices of the balls predicted path, starting with where it hit the paddle,
    #and a coordinate for each ball bounce, ending with where it hits the opponent's field
    trajectoryVertices = [(ball_posX, worldspace_bottom)]
    
    bounces = 0
    
    #Key variables for the Machine Learning AI
    #Whenever the game is reset or a collision with a paddle occurs,
    #i.e. the ball starts moving to the other side of the field, the position & velocity at that time are stored
    lastHit_posX = ball_posX
    lastHit_posY = ball_posY
    lastHit_vX = ball_vX
    lastHit_vY = ball_vY
        
    def __init__(self):
        pass
    
    def saveLastHit(self):
        self.lastHit_posX = self.ball_posX
        self.lastHit_posY = self.ball_posY
        self.lastHit_vX = self.ball_vX
        self.lastHit_vY = self.ball_vY
    
    def resetGame(self, posX, direction):
        self.ball_posX = posX
        self.ball_posY = self.worldspace_top + self.worldspace_height / 2
        self.ball_radius = 0.025
        self.ball_unitsPerTick = self.ball_unitsPerTick_default
        
        game.ball_vY = game.ball_unitsPerTick * (random.uniform(-0.5,0.5))
        #Ensure the total speed per ms is equal to unitsPerTick
        if (direction > 0):
            game.ball_vX = (abs(game.ball_unitsPerTick**2 - game.ball_vY**2))**0.5
        else:
            game.ball_vX = -(abs(game.ball_unitsPerTick**2 - game.ball_vY**2))**0.5
                
        self.paddle_posX_p1 = self.worldspace_left
        self.paddle_posY_p1 = self.worldspace_top + self.worldspace_height / 2 - self.paddle_height / 2 
        self.paddle_posX_p2 = self.worldspace_right - self.paddle_width
        self.paddle_posY_p2 = self.worldspace_top + self.worldspace_height / 2 - self.paddle_height / 2
        
        self.trajectoryVertices = [(self.ball_posX, self.worldspace_bottom)]
        self.bounces = 0
        self.saveLastHit()
        game.predictTrajectory(int(1000/60))
    
    def moveP1paddle(self, dt):
        self.paddle_posY_p1 += self.paddle_unitsPerTick * dt
        self.paddle_posY_p1 = min(self.worldspace_bottom-self.paddle_height, max(self.worldspace_top, self.paddle_posY_p1))
        
    def moveP2paddle(self, dt):
        self.paddle_posY_p2 += self.paddle_unitsPerTick * dt
        self.paddle_posY_p2 = min(self.worldspace_bottom-self.paddle_height, max(self.worldspace_top, self.paddle_posY_p2))

    def worldToScreenCoord(self, coord):
        return int(coord * self.pixels_per_unit)
    
    def screenToWorldCoord(self, coord):
        return coord / (self.pixels_per_unit)
    
    """Calculates the game logic per frame.
    - Parameter "dt" is the time passed in milliseconds, governing how much the ball moves ahead given it's velocity, i.e.
      updating it's position
    - If there is a collision with a paddle, the ball bounces back, i.e. it's velocity is updated;
      the position & velocity after impact is stored (for the Machine Learning AI); and the trajectory is predicted (for the hardcoded AI)
    - If however the ball moves past the paddle, the gamestate is updated to either player 1 or 2 winning
    """
    def updateBallPos(self, dt):
        self.ball_posX += self.ball_vX * dt
        self.ball_posY += self.ball_vY * dt
        
        #Check collision with right paddle
        if (self.ball_posX > self.worldspace_width - self.paddle_width):
            if (self.paddle_posY_p2 - self.ball_radius  < self.ball_posY < self.ball_radius + self.paddle_posY_p2 + self.paddle_height):
                #Collision occurred, update
                self.bounces += 1
                self.ball_posX = self.worldspace_right - self.paddle_width
                #The y-direction of the ball (exit angle) is dictated by the distance to the paddle's middle
                self.ball_vY = self.ball_unitsPerTick * 0.98*(self.ball_posY - self.paddle_posY_p2 - self.paddle_height / 2 ) / (self.paddle_height / 2 + self.ball_radius)
                #Ensure the total velocity vector magnitude is equal to unitsPerTick
                self.ball_vX = -(abs(self.ball_unitsPerTick**2 - self.ball_vY**2))**0.5
                self.ball_unitsPerTick += 0.0002/math.log2(self.bounces+1)
                self.predictTrajectory(dt)
                self.saveLastHit()
            elif (self.ball_posX > self.worldspace_width):
                #Ball passed the right paddle
                self.state = GameState.P1WON
                
            
        #Check collision with with left paddle
        if (self.ball_posX < self.paddle_width):
            if (self.paddle_posY_p1 - self.ball_radius  < self.ball_posY < self.ball_radius + self.paddle_posY_p1 + self.paddle_height):
                #Collision occurred, update
                self.bounces += 1
                self.ball_posX = self.worldspace_left + self.paddle_width
                #The y-direction of the ball (exit angle) is dictated by the distance to the paddle's middle
                self.ball_vY = self.ball_unitsPerTick * 0.98*(self.ball_posY - self.paddle_posY_p1 - self.paddle_height / 2 ) / (self.paddle_height / 2 + self.ball_radius)
                #Ensure the total velocity vector magnitude is equal to unitsPerTick
                self.ball_vX = (abs(self.ball_unitsPerTick**2 - self.ball_vY**2))**0.5
                self.ball_unitsPerTick += 0.0002/math.log2(self.bounces+1)
                self.predictTrajectory(dt)
                self.saveLastHit()
            elif (self.ball_posX < 0 ):
                #Ball passed the left paddle
                self.state = GameState.P2WON
                
        #Check collision with top wall
        if (self.ball_posY - self.ball_radius < self.worldspace_top):
            self.ball_posY = self.worldspace_top + self.ball_radius
            self.ball_vY *= -1   

        #Check collision with bottom wall
        if (self.ball_posY + self.ball_radius > self.worldspace_bottom):
            self.ball_posY = self.worldspace_bottom - self.ball_radius
            self.ball_vY *= -1
    
    """Predicts the ball's trajectory until point of impact
    Stores all points of impact with the walls, as well as the final point of impact in the targeted player's court,
    inside trajectoryVertices. Parameter dt is the precision with which the trajectory is simulated, measured
    in milliseconds between updating the world's coordinate system"""
    def predictTrajectory(self, dt):
        
        ball_posX = self.ball_posX
        ball_posY = self.ball_posY
        ball_vX = self.ball_vX
        ball_vY = self.ball_vY
        
        self.trajectoryVertices = [(ball_posX,ball_posY)]
        
        while ((ball_posX <= self.worldspace_width - self.paddle_width)) and (ball_posX >= self.paddle_width):
            ball_posX += ball_vX * dt
            ball_posY += ball_vY * dt
            
            #Check collision with top wall
            if (ball_posY - self.ball_radius < self.worldspace_top):
                ball_posY = self.worldspace_top + self.ball_radius
                ball_vY *= -1    
                self.trajectoryVertices.append((ball_posX,ball_posY))
                
            #Check collision with bottom wall
            if (ball_posY + self.ball_radius > self.worldspace_bottom):
                ball_posY = self.worldspace_bottom - self.ball_radius
                ball_vY *= -1
                self.trajectoryVertices.append((ball_posX,ball_posY))
                
        self.trajectoryVertices.append((ball_posX,ball_posY))
                

"""Encapsulates the Machine Learning Algorithms for the right paddle (player 2)"""
class rightPaddleAI_ML:
    #Neural network input training data: Velocity vector and y-position on the right hand of the screen
    trainingSamples_v = []
    trainingSamples_pos = []
    #Neural network output training data: y-position of collision on the AI's court
    trainingSamples_collisionPosY = []
    
    #Model hyperparameters hardcoded here
    model = Sequential([
            Dense(units=128, input_shape=(4,), activation='relu'),
            Dense(units=64, activation='relu'),
            Dense(units=64, activation='relu'),
            Dense(units=1)
        ])
    gamesTrained = 0
    
    def __init__(self):
        self.model.summary()
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    
    """Generate 'count' number of ball throws with random starting positions and velocities"""
    def generateTrainingData(self, game, count, dt):
        for i in range(count):
            game.resetGame(game.worldspace_left + game.paddle_width, game.ball_unitsPerTick_default)
            game.ball_posY = random.uniform(game.worldspace_top + game.ball_radius, game.worldspace_bottom - game.ball_radius)
            game.ball_vY = game.ball_unitsPerTick * (random.uniform(-0.98,0.98))
            game.ball_vX = (abs(game.ball_unitsPerTick**2 - game.ball_vY**2))**0.5
            game.predictTrajectory(dt)
            self.trainingSamples_v.append([game.ball_vX/game.ball_unitsPerTick, game.ball_vY/game.ball_unitsPerTick])
            self.trainingSamples_pos.append([game.ball_posX, game.ball_posY])
            self.trainingSamples_collisionPosY.append(game.trajectoryVertices[-1][1])
            
    """Train the model using trainingSamples_-variables generated before. Training hyperparameters hardcoded here"""
    def train(self):
        train_samples = np.concatenate((self.trainingSamples_pos, self.trainingSamples_v), axis=1)
        self.model.fit(x = train_samples, y=np.array(self.trainingSamples_collisionPosY), validation_split=0.0, epochs=15, batch_size = 128, shuffle=False, verbose=2)
        self.gamesTrained += len(self.trainingSamples_collisionPosY)
        self.trainingSamples_pos = []
        self.trainingSamples_v = []
        self.trainingSamples_collisionPosY = []
        
    """Make the model predict the impact y-position based on an (x,y) ball coordinate and velocity vector"""
    def predict(self, posX, posY, vX, vY):
        input_v = np.array([vX,vY]).reshape((-1,2))
        input_pos = np.array([posX,posY]).reshape((-1,2))
        input = np.concatenate((input_pos, input_v), axis=1)
        return self.model.predict(input)[0]
    

"""Class to get a blinking score effect for a set amount of frames"""
class blinkingText:

    framesBlinked = 0
    
    #Blink duration in frames
    def __init__(self, posX, posY, text, blinkDur):
        self.posX = posX
        self.posY = posY
        self.text = text
        self.blinkDur = blinkDur
        
    def render(self, screen, font_text):
        self.framesBlinked += 1
        if (self.framesBlinked < self.blinkDur):
            if (self.framesBlinked % 20 < 10):
                srf = font_text.render(str(self.text), True, red)
            else:
                srf = font_text.render(str(self.text), True, darkred)
            screen.blit(srf, (self.posX, self.posY))
    
    
import pygame


pygame.init()
clock = pygame.time.Clock()
random.seed(clock.get_time())

game = Pong()
game.resetGame(game.worldspace_left + game.paddle_width, +1)
running = True
gamePaused = True

screen = pygame.display.set_mode((int(game.pixels_per_unit * (game.worldspace_right - game.worldspace_left)),
                                  int(game.pixels_per_unit * (game.worldspace_bottom - game.worldspace_top) + 210)))
pygame.display.set_caption('PONG')
font_text = pygame.font.SysFont("couriernew", 20)
scoreText = blinkingText(game.worldToScreenCoord(game.worldspace_width/2) - 35, 40, "Score!", 60)
scoreText.framesBlinked = 60

"""AI/NN SETUP"""
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
print(physical_devices)
print(tensorflow.__version__)
print(tensorflow.test.is_built_with_cuda())
ml_ai = rightPaddleAI_ML()
p2PlayerType = PlayerType.ML_AI
p2AI_randmax = game.paddle_height/2
p2AI_randvalue = random.uniform(-p2AI_randmax, p2AI_randmax)
displayPrediction = True


"""MAIN GAME LOOP"""
while running:
    dt = clock.tick(60)
    if (random.randrange(0,1000) < dt):
        p2AI_randvalue = random.uniform(-p2AI_randmax, p2AI_randmax)
        
    """CATCH AND HANDLE INPUTS"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                gamePaused = not gamePaused
            if event.key == pygame.K_t:
                ml_ai.generateTrainingData(game, 1000, dt)
                ml_ai.train()
                game.resetGame(game.worldspace_right - game.paddle_width, -1)
            if event.key == pygame.K_r:
                p2PlayerType = (p2PlayerType + 1) % 3
            if event.key == pygame.K_d:
                displayPrediction = not displayPrediction
            if event.key == pygame.K_o:
                game.score_p1 = 0
                game.score_p2 = 0
        keys_pressed = pygame.key.get_pressed()

    """PLAYER CONTROL"""
    if (not gamePaused):
        if keys_pressed[pygame.K_UP]:
            game.moveP1paddle(-dt)
        if keys_pressed[pygame.K_DOWN]:
            game.moveP1paddle(dt)
        if keys_pressed[pygame.K_w]:
            game.moveP2paddle(-dt)
        if keys_pressed[pygame.K_s]:
            game.moveP2paddle(dt)
        
    """GAME LOGIC"""
    if (not gamePaused):
        game.updateBallPos(dt)
        
    if (game.state == GameState.P1WON):
        game.score_p1 += 1
        game.resetGame(game.worldspace_left + game.paddle_width, 1)
        game.state = GameState.RUNNING
        scoreText.framesBlinked = 0 
    elif (game.state == GameState.P2WON):
        game.score_p2 += 1
        game.resetGame(game.worldspace_right - game.paddle_width, -1)
        game.state = GameState.RUNNING
        scoreText.framesBlinked = 0 
        
    """AI CONTROL"""
    if (not gamePaused):
        if (p2PlayerType == PlayerType.ML_AI):
            if (game.ball_vX > 0):
                pred_collision = ml_ai.predict(game.lastHit_posX, game.lastHit_posY, game.lastHit_vX/game.ball_unitsPerTick, game.lastHit_vY/game.ball_unitsPerTick)
                destY_p1 =  pred_collision - game.paddle_height / 2
                if (destY_p1 < game.paddle_posY_p2 - 0.5*dt*game.paddle_unitsPerTick):
                    game.moveP2paddle(-dt)
                elif (destY_p1 > game.paddle_posY_p2 + 0.5*dt*game.paddle_unitsPerTick):
                    game.moveP2paddle(dt)
        elif (p2PlayerType == PlayerType.HARDCODED_AI):
            if (game.ball_vX > 0):
                destY = game.trajectoryVertices[-1][1] - game.paddle_height / 2 + p2AI_randvalue
            else:
                destY = game.worldspace_height/2 - game.paddle_height/2
            if (destY < game.paddle_posY_p2 - 0.5*dt*game.paddle_unitsPerTick):
                game.moveP2paddle(-dt)
            elif (destY > game.paddle_posY_p2 + 0.5*dt*game.paddle_unitsPerTick):
                game.moveP2paddle(dt)

 
    """DRAW GAME OBJECTS"""
    screen.fill(black)
    #Draw both paddles and the ball
    pygame.draw.rect(screen, white, (game.worldToScreenCoord(game.paddle_posX_p1), 
                                     game.worldToScreenCoord(game.paddle_posY_p1), 
                                     game.worldToScreenCoord(game.paddle_width), 
                                     game.worldToScreenCoord(game.paddle_height)))
    
    pygame.draw.rect(screen, white, (game.worldToScreenCoord(game.paddle_posX_p2), 
                                     game.worldToScreenCoord(game.paddle_posY_p2), 
                                     game.worldToScreenCoord(game.paddle_width), 
                                     game.worldToScreenCoord(game.paddle_height)))
     
    pygame.draw.circle(screen, white, (game.worldToScreenCoord(game.ball_posX), 
                                       game.worldToScreenCoord(game.ball_posY)), 
                                       game.worldToScreenCoord(game.ball_radius), 0)
    
    #Draw field separator
    lineCount = 15
    for i in range(1,lineCount*2,2):
        pygame.draw.line(screen, white, (game.worldToScreenCoord(game.worldspace_left + game.worldspace_width/2),
                                         game.worldToScreenCoord(game.worldspace_top + (i-0.5)*game.worldspace_height/(2*lineCount))),
                                        ((game.worldToScreenCoord(game.worldspace_left + game.worldspace_width/2),
                                         game.worldToScreenCoord(game.worldspace_top + (i+0.5)*game.worldspace_height/(2*lineCount)))))
    if (not gamePaused):
        if (displayPrediction):
            pygame.draw.circle(screen, red, (game.worldToScreenCoord(game.trajectoryVertices[-1][0]),
                                         game.worldToScreenCoord(game.trajectoryVertices[-1][1])),
                                         game.worldToScreenCoord(game.ball_radius))
            if (game.ball_vX > 0):
                pygame.draw.circle(screen, blue, (game.worldToScreenCoord(game.worldspace_width - game.paddle_width),
                             game.worldToScreenCoord(pred_collision)),
                             game.worldToScreenCoord(game.ball_radius))
            
    #Lower boundary
    pygame.draw.line(screen, white, (game.worldToScreenCoord(game.worldspace_left),
                                         game.worldToScreenCoord(game.worldspace_bottom)),
                                        ((game.worldToScreenCoord(game.worldspace_right),
                                         game.worldToScreenCoord(game.worldspace_bottom))))
    
    #Draw text
    srf = font_text.render(str(game.score_p1), True, white)
    screen.blit(srf, (game.worldToScreenCoord(game.worldspace_left +game.worldspace_width/2)-30, game.worldToScreenCoord(0)))
    srf = font_text.render(str(game.score_p2), True, white)
    screen.blit(srf, (game.worldToScreenCoord(game.worldspace_left +game.worldspace_width/2)+20, game.worldToScreenCoord(0)))
    
    srf = font_text.render(str(game.bounces), True, white)
    screen.blit(srf, (game.worldToScreenCoord(game.worldspace_right)-30, game.worldToScreenCoord(0)))
    
    srf = font_text.render("UP/DOWN keys to move paddle", True, white)
    screen.blit(srf, (game.worldToScreenCoord(0)+10, game.worldToScreenCoord(game.worldspace_bottom)+0))
    srf = font_text.render("Training set size: " + str(ml_ai.gamesTrained), True, white)
    screen.blit(srf, (game.worldToScreenCoord(0)+10, game.worldToScreenCoord(game.worldspace_bottom)+20))
    srf = font_text.render("Keys:", True, white)
    screen.blit(srf, (game.worldToScreenCoord(0)+10, game.worldToScreenCoord(game.worldspace_bottom)+40))
    srf = font_text.render("  t: Train on 1000 games", True, white)
    screen.blit(srf, (game.worldToScreenCoord(0)+10, game.worldToScreenCoord(game.worldspace_bottom)+60))
    srf = font_text.render("  r: Toggle right paddle player", True, white)
    screen.blit(srf, (game.worldToScreenCoord(0)+10, game.worldToScreenCoord(game.worldspace_bottom)+80))
    if (p2PlayerType == PlayerType.HARDCODED_AI):
        srf = font_text.render("    (current: Hardcoded AI)", True, white)
    elif (p2PlayerType == PlayerType.HUMAN):
        srf = font_text.render("    (current: Human, controls: w/s)", True, white)
    else:
        srf = font_text.render("    (current: Machine Learning AI)", True, white)
    screen.blit(srf, (game.worldToScreenCoord(0)+10, game.worldToScreenCoord(game.worldspace_bottom)+100))
    srf = font_text.render("  p: Pause game", True, white)
    screen.blit(srf, (game.worldToScreenCoord(0)+10, game.worldToScreenCoord(game.worldspace_bottom)+120))
    srf = font_text.render("  o: Reset score", True, white)
    screen.blit(srf, (game.worldToScreenCoord(0)+10, game.worldToScreenCoord(game.worldspace_bottom)+140))
    srf = font_text.render("  d: Toggle display of projected impact", True, white)
    screen.blit(srf, (game.worldToScreenCoord(0)+10, game.worldToScreenCoord(game.worldspace_bottom)+160))
    srf = font_text.render("    (blue:ml, red:hardcoded)", True, white)
    screen.blit(srf, (game.worldToScreenCoord(0)+10, game.worldToScreenCoord(game.worldspace_bottom)+180))
    
    scoreText.render(screen, font_text)
    
    if (gamePaused):
        srf = font_text.render("PAUSED", True, white)
        screen.blit(srf, (game.worldToScreenCoord(game.worldspace_width/2)-30, game.worldToScreenCoord(game.worldspace_height/2)))
        
    pygame.display.update()
    
pygame.quit()