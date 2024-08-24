import tkinter as tk
from tkinter import messagebox
import random

# Class for the Hangman game
class HangmanGame(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Hangman Game")
        self.center_window(800, 600)  # Center the window
        self.pack()
        self.create_widgets()
        self.new_game()

    def create_widgets(self):
        # Create a label to show the word with underscores
        self.word_label = tk.Label(self, text="", font=("Helvetica", 24))
        self.word_label.pack(pady=20)

        # Create a label to show the number of incorrect guesses left
        self.incorrect_guesses_label = tk.Label(self, text="", font=("Helvetica", 18))
        self.incorrect_guesses_label.pack(pady=10, anchor='ne')

        # Bind the key press event to the guess_letter method
        self.master.bind("<KeyPress>", self.on_key_press)

        # Create a button to start a new game
        self.new_game_button = tk.Button(self, text="New Game", command=self.new_game)
        self.new_game_button.pack(pady=20)

    def new_game(self):
        self.word = random.choice(self.get_word_pool())
        self.correct_letters = set(self.word)
        self.guessed_letters = set()
        self.incorrect_guesses = 0
        self.max_incorrect_guesses = 3
        self.update_display()

    def get_word_pool(self):
        return [
            "PYTHON", "JAVA", "JAVASCRIPT", "HANGMAN", "PROGRAMMING", "DEVELOPER", "SOFTWARE", "COMPUTER",
            "ENGINEERING", "DATABASE", "NETWORKING", "ALGORITHM", "FUNCTION", "VARIABLE", "DEBUGGING", "SYNTAX",
            "COMPILER", "INTERPRETER", "OPERATING", "SYSTEM", "MEMORY", "STORAGE", "VIRTUALIZATION", "CLOUD",
            "SERVER", "CLIENT", "PROTOCOL", "INTERFACE", "MODULAR", "OBJECT", "ORIENTED", "DESIGN", "ARCHITECTURE",
            "COMPONENT", "MICROSERVICE", "CONTROLLER", "MODEL", "VIEW", "TEMPLATE", "REPOSITORY", "DEPENDENCY",
            "INJECTION", "AUTOMATION", "INTEGRATION", "DELIVERY", "PIPELINE", "VERSION", "CONTROL", "GIT", "DOCKER",
            "KUBERNETES", "CONTAINER", "ORCHESTRATION", "SCALABILITY", "PERFORMANCE", "OPTIMIZATION", "LOAD",
            "BALANCING", "CACHE", "REDIS", "SECURITY", "ENCRYPTION", "AUTHENTICATION", "AUTHORIZATION", "SESSION",
            "COOKIE", "TOKEN", "FIREWALL", "INTRUSION", "DETECTION", "PREVENTION", "MACHINE", "LEARNING", "ARTIFICIAL",
            "INTELLIGENCE", "DATA", "SCIENCE", "ANALYTICS", "VISUALIZATION", "STATISTICS", "PROBABILITY", "REGRESSION",
            "CLASSIFICATION", "CLUSTERING", "NEURAL", "NETWORK", "DEEP", "LEARNING", "NATURAL", "LANGUAGE", "PROCESSING",
            "COMPUTER", "VISION", "ROBOTICS", "AUTOMATION", "SIMULATION", "ANIMATION", "VIRTUAL", "REALITY", "AUGMENTED"
        ]

    def update_display(self):
        display_word = [letter if letter in self.guessed_letters else "_" for letter in self.word]
        self.word_label.config(text=" ".join(display_word))
        self.incorrect_guesses_label.config(text=f"Incorrect Guesses Left: {self.max_incorrect_guesses - self.incorrect_guesses}")
        
        if "_" not in display_word:
            self.show_popup("You win!", "Congratulations!")
        elif self.incorrect_guesses >= self.max_incorrect_guesses:
            self.show_game_over_popup()

    def on_key_press(self, event):
        letter = event.char.upper()
        if letter.isalpha() and letter not in self.guessed_letters:
            self.guessed_letters.add(letter)
            if letter not in self.correct_letters:
                self.incorrect_guesses += 1
            self.update_display()

    def show_popup(self, message, title):
        popup = tk.Toplevel(self)
        popup.title(title)
        popup.geometry("300x150")
        popup.grab_set()  # Make the popup modal
        popup.update_idletasks()  # Update geometry before centering

        # Center the popup
        x = self.master.winfo_x() + (self.master.winfo_width() - popup.winfo_width()) // 2
        y = self.master.winfo_y() + (self.master.winfo_height() - popup.winfo_height()) // 2
        popup.geometry(f"+{x}+{y}")

        tk.Label(popup, text=message, font=("Helvetica", 14)).pack(pady=20)
        tk.Button(popup, text="Restart Game", command=lambda: [popup.destroy(), self.new_game()]).pack()

    def show_game_over_popup(self):
        popup = tk.Toplevel(self)
        popup.title("Game Over")
        popup.geometry("400x200")  # Increased the size to accommodate the message
        popup.grab_set()  # Make the popup modal
        popup.update_idletasks()  # Update geometry before centering

        # Center the popup
        x = self.master.winfo_x() + (self.master.winfo_width() - popup.winfo_width()) // 2
        y = self.master.winfo_y() + (self.master.winfo_height() - popup.winfo_height()) // 2
        popup.geometry(f"+{x}+{y}")

        # Message to be displayed in the popup
        message = f"You lose!\nThe word was {self.word}"
        tk.Label(popup, text=message, font=("Helvetica", 14)).pack(pady=20)
        button_frame = tk.Frame(popup)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Restart", command=lambda: [popup.destroy(), self.new_game()]).pack(side="left", padx=10)
        tk.Button(button_frame, text="Quit", command=self.master.quit).pack(side="right", padx=10)

    def center_window(self, width, height):
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.master.geometry(f"{width}x{height}+{x}+{y}")

# Main function to run the Hangman game
def main():
    root = tk.Tk()
    game = HangmanGame(master=root)
    root.mainloop()

if __name__ == "__main__":
    main()
