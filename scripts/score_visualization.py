#!/usr/bin/env python3.6
import rospy
import pygame as pg
from human_robot_collaborative_learning.msg import Score

class ShowScore():
    def __init__(self):
        sub = rospy.Subscriber('score_topic', Score, self.score_callback)
        pg.init()
        pg.font.init()
        self.display = (800, 600)
        self.screen = pg.display.set_mode(self.display)
        pg.display.set_caption("Game Score Display")
        
        # Different font sizes for different elements
        self.title_font = pg.font.SysFont('arial', 24, bold=True)
        self.header_font = pg.font.SysFont('arial', 32, bold=True)
        self.score_font = pg.font.SysFont('arial', 48, bold=True)
        self.status_font = pg.font.SysFont('arial', 36, bold=True)
        
        self.outcome = None
        self.msg_count, self.clean_screen_count = 1, 0
        self.running = True
        
        # New variables for tracking games and wins
        self.current_game = 0
        self.max_games = 8
        self.current_block = 1
        self.max_blocks = 15
        
        # Array to store wins for each block (index 0 unused, blocks 1-17)
        self.wins_per_block = [0] * (self.max_blocks + 1)
        self.current_block_wins = 0
        self.avg_wins_previous_blocks = 0.0
        
        # Timer variables
        self.timer_start = pg.time.get_ticks()
        self.countdown_duration = 30  # 30 seconds
        self.time_remaining = self.countdown_duration
        self.timer_running = False  # Timer starts stopped

    def calculate_average_wins_previous_blocks(self, current_block_num):
        """Calculate average wins from previous blocks"""
        if current_block_num <= 1:
            return 0.0
        
        # Sum wins from blocks 1 to current_block_num-1
        total_wins = sum(self.wins_per_block[1:current_block_num])
        num_previous_blocks = current_block_num - 1
        
        return total_wins / num_previous_blocks if num_previous_blocks > 0 else 0.0

    def run(self):
        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_q):
                    self.running = False
            
            # Update countdown timer only if timer is running
            current_time = pg.time.get_ticks()
            if self.timer_running:
                elapsed_time = (current_time - self.timer_start) / 1000.0  # Convert to seconds
                self.time_remaining = max(0, self.countdown_duration - elapsed_time)
            # If timer is not running, keep the current time_remaining value
            
            # Clear screen every frame to prevent visual artifacts
            self.screen.fill((20, 30, 40))  # Dark blue background
            
            # Draw decorative border
            pg.draw.rect(self.screen, (70, 130, 180), (10, 10, self.display[0]-20, self.display[1]-20), 3)
            
            # TOP SECTION - Block and Game Info
            top_y = 40
            
            # Block number (x / 17)
            block_label = self.title_font.render("BLOCK", False, (150, 150, 150))
            #block_value = self.header_font.render(f"{self.current_block} / {self.max_blocks}", False, (100, 200, 255))
            block_value = self.header_font.render(f"{self.current_block}", False, (100, 200, 255))
            block_label_x = self.display[0] // 4 - block_label.get_width() // 2
            block_value_x = self.display[0] // 4 - block_value.get_width() // 2
            self.screen.blit(block_label, (block_label_x, top_y))
            self.screen.blit(block_value, (block_value_x, top_y + 35))
            
            # Game number (x / 8)
            game_label = self.title_font.render("EPISODE", False, (150, 150, 150))
            game_value = self.header_font.render(f"{self.current_game} / {self.max_games}", False, (100, 200, 255))
            game_label_x = 3 * self.display[0] // 4 - game_label.get_width() // 2
            game_value_x = 3 * self.display[0] // 4 - game_value.get_width() // 2
            self.screen.blit(game_label, (game_label_x, top_y))
            self.screen.blit(game_value, (game_value_x, top_y + 35))
            
            # Separator line
            pg.draw.line(self.screen, (70, 130, 180), (50, 140), (self.display[0]-50, 140), 2)
            
            # CENTER SECTION - Wins
            center_y = 200
            
            # Current Block Wins
            current_wins_label = self.title_font.render("WINS ON CURRENT BLOCK", False, (150, 150, 150))
            current_wins_value = self.score_font.render(str(self.current_block_wins), False, (255, 255, 255))
            current_wins_label_x = self.display[0] // 2 - current_wins_label.get_width() // 2
            current_wins_value_x = self.display[0] // 2 - current_wins_value.get_width() // 2
            self.screen.blit(current_wins_label, (current_wins_label_x, center_y))
            self.screen.blit(current_wins_value, (current_wins_value_x, center_y + 35))
            
            # Average Wins in Previous Blocks
            avg_wins_label = self.title_font.render("AVERAGE WINS IN PREVIOUS BLOCKS", False, (150, 150, 150))
            avg_wins_value = self.score_font.render(f"{self.avg_wins_previous_blocks:.1f}", False, (255, 215, 0))  # Gold color
            avg_wins_label_x = self.display[0] // 2 - avg_wins_label.get_width() // 2
            avg_wins_value_x = self.display[0] // 2 - avg_wins_value.get_width() // 2
            self.screen.blit(avg_wins_label, (avg_wins_label_x, center_y + 120))
            self.screen.blit(avg_wins_value, (avg_wins_value_x, center_y + 155))
            
            # Add a subtle glow effect around average wins if it's greater than 0
            if self.avg_wins_previous_blocks > 0:
                glow_rect = pg.Rect(avg_wins_value_x - 10, center_y + 145, 
                                   avg_wins_value.get_width() + 20, avg_wins_value.get_height() + 20)
                pg.draw.rect(self.screen, (255, 215, 0, 30), glow_rect, 2)
            
            # BOTTOM SECTION - Animated Countdown Timer
            timer_y = 450
            timer_label = self.title_font.render("TIME", False, (150, 150, 150))
            timer_label_x = self.display[0] // 2 - timer_label.get_width() // 2
            self.screen.blit(timer_label, (timer_label_x, timer_y))
            
            # Timer display with color changes based on remaining time
            if self.time_remaining > 10:
                timer_color = (100, 200, 255)  # Blue for plenty of time
            elif self.time_remaining > 5:
                timer_color = (255, 200, 0)    # Orange for warning
            else:
                timer_color = (255, 80, 80)    # Red for critical
            
            timer_text = self.status_font.render(f"{self.time_remaining:.1f}s", False, timer_color)
            timer_text_x = self.display[0] // 2 - timer_text.get_width() // 2
            self.screen.blit(timer_text, (timer_text_x, timer_y + 35))
            
            # Animated circular progress bar for timer
            center_x = self.display[0] // 2
            center_y = timer_y + 50
            radius = 80
            
            # Background circle (dark gray)
            pg.draw.circle(self.screen, (60, 60, 60), (center_x, center_y), radius, 4)
            
            # Progress arc
            if self.time_remaining > 0:
                progress = self.time_remaining / self.countdown_duration
                
                # Determine arc color based on remaining time
                if progress > 0.33:
                    arc_color = (100, 200, 255)  # Blue
                elif progress > 0.17:
                    arc_color = (255, 200, 0)    # Orange
                else:
                    arc_color = (255, 80, 80)    # Red
                
                # Draw progress arc using pygame's arc function
                import math
                start_angle = -math.pi / 2  # Start at 12 o'clock
                end_angle = start_angle + (2 * math.pi * progress)
                
                # Draw the arc more simply to avoid overlap issues
                if progress > 0.01:  # Only draw if there's meaningful progress
                    # Convert angles to degrees for pygame (pygame uses degrees for arc)
                    start_deg = math.degrees(start_angle)
                    end_deg = math.degrees(end_angle)
                    
                    # Create a rectangle for the arc
                    arc_rect = pg.Rect(center_x - radius, center_y - radius, radius * 2, radius * 2)
                    
                    # Draw the arc
                    pg.draw.arc(self.screen, arc_color, arc_rect, start_angle, end_angle, 6)
            
            # Pulsing effect when time is running out
            if self.time_remaining <= 5 and self.time_remaining > 0:
                pulse_intensity = int(50 + 50 * abs(math.sin(current_time / 200)))
                pulse_color = (255, pulse_intensity, pulse_intensity)
                pg.draw.circle(self.screen, pulse_color, (center_x, center_y), radius + 8, 2)
            
            pg.display.update()

    def score_callback(self, msg):
        self.msg_count += 1
        score = msg.score.data
        new_block = msg.block.data
        status = msg.status.data
        
        if new_block < 1:
            return
        
        # Check if we moved to a new block
        if new_block != self.current_block:
            # Save wins for the previous block (if it was valid)
            if self.current_block > 0:
                self.wins_per_block[self.current_block] = self.current_block_wins
            
            # Update to new block
            self.current_block = new_block
            self.current_block_wins = 0
            self.current_game = 0
            
            # Update average wins for previous blocks
            self.avg_wins_previous_blocks = self.calculate_average_wins_previous_blocks(self.current_block)
        
        # Check if current score is a win (score > 0)
        if score > 0:
            self.current_block_wins += 1

        # Handle timer and game logic based on status
        if status == "Start":
            # Reset timer and start new game
            self.timer_start = pg.time.get_ticks()
            self.time_remaining = self.countdown_duration
            self.timer_running = True
            self.current_game = min(self.current_game + 1, self.max_games)
        elif status == "Success" or status == "Timeout":
            # Stop timer
            self.timer_running = False
    
def main():
    rospy.init_node("score_visualization")
    show_score = ShowScore()
    show_score.run()

if __name__ == "__main__":
    main()