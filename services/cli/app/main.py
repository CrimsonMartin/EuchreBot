"""
Euchre CLI - Terminal-based game interface
"""

import os
import sys
from colorama import init, Fore, Style
from euchre_core import EuchreGame, PlayerType, Card, Suit, GamePhase

# Initialize colorama
init(autoreset=True)

SUITS_DISPLAY = {
    'C': f'{Fore.GREEN}♣{Style.RESET_ALL}',
    'D': f'{Fore.RED}♦{Style.RESET_ALL}',
    'H': f'{Fore.RED}♥{Style.RESET_ALL}',
    'S': f'{Fore.GREEN}♠{Style.RESET_ALL}',
}


class EuchreCLI:
    def __init__(self):
        self.game = None
        
    def clear_screen(self):
        os.system('clear' if os.name != 'nt' else 'cls')
    
    def print_banner(self):
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}                    🃏 EUCHRE BOT 🃏{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    def format_card(self, card: Card) -> str:
        """Format a card for display"""
        suit_symbol = SUITS_DISPLAY.get(card.suit.value, card.suit.value)
        return f"{card.rank}{suit_symbol}"
    
    def display_scoreboard(self):
        """Display current scores"""
        state = self.game.state
        print(f"\n{Fore.YELLOW}╔═══════════ SCOREBOARD ═══════════╗{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  Team 1 (P0 & P2): {state.team1_score:2d}  |  Team 2 (P1 & P3): {state.team2_score:2d}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}╚═══════════════════════════════════╝{Style.RESET_ALL}\n")
    
    def display_game_state(self, player_position: int = 0):
        """Display current game state"""
        state = self.game.state
        player = state.players[player_position]
        
        self.clear_screen()
        self.print_banner()
        self.display_scoreboard()
        
        # Display current phase
        print(f"{Fore.MAGENTA}Phase: {state.phase.value}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Hand: {state.hand_number}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Dealer: Player {state.dealer_position}{Style.RESET_ALL}\n")
        
        # Display trump info if set
        if state.trump:
            trump_symbol = SUITS_DISPLAY.get(state.trump.value, state.trump.value)
            print(f"{Fore.GREEN}Trump: {trump_symbol}{Style.RESET_ALL}")
            if state.trump_caller_position is not None:
                print(f"{Fore.GREEN}Called by: Player {state.trump_caller_position}{Style.RESET_ALL}")
            if state.going_alone:
                print(f"{Fore.YELLOW}Going Alone!{Style.RESET_ALL}")
            print()
        
        # Display turned up card during trump selection
        if state.turned_up_card and not state.trump:
            card_display = self.format_card(state.turned_up_card)
            print(f"{Fore.CYAN}Turned up card: {card_display}{Style.RESET_ALL}\n")
        
        # Display current trick
        if state.current_trick and state.current_trick.cards:
            print(f"{Fore.YELLOW}Current Trick:{Style.RESET_ALL}")
            for pos, card in state.current_trick.cards:
                card_display = self.format_card(card)
                print(f"  Player {pos}: {card_display}")
            print()
        
        # Display player's hand
        print(f"{Fore.CYAN}Your Hand (Player {player_position}):{Style.RESET_ALL}")
        hand_str = "  ".join(f"[{i}] {self.format_card(card)}" for i, card in enumerate(player.hand))
        print(f"  {hand_str}\n")
        
        # Display whose turn it is
        current_player_name = state.players[state.current_player_position].name
        if state.current_player_position == player_position:
            print(f"{Fore.GREEN}>>> YOUR TURN <<<{Style.RESET_ALL}\n")
        else:
            print(f"{Fore.YELLOW}Waiting for {current_player_name} (P{state.current_player_position})...{Style.RESET_ALL}\n")
    
    def setup_game(self):
        """Setup a new game"""
        self.clear_screen()
        self.print_banner()
        
        print(f"{Fore.YELLOW}Setting up new game...{Style.RESET_ALL}\n")
        
        # Create game
        self.game = EuchreGame("cli-game-001")
        
        # Add players
        self.game.add_player("You", PlayerType.HUMAN)
        self.game.add_player("AI Player 1", PlayerType.RANDOM_AI)
        self.game.add_player("AI Player 2", PlayerType.RANDOM_AI)
        self.game.add_player("AI Player 3", PlayerType.RANDOM_AI)
        
        print(f"{Fore.GREEN}Players added successfully!{Style.RESET_ALL}")
        print(f"  Player 0 (You) vs Player 1 (AI)")
        print(f"  Player 2 (AI) vs Player 3 (AI)\n")
        
        input("Press Enter to start the game...")
        
        # Start first hand
        self.game.start_new_hand()
    
    def handle_trump_selection(self, player_position: int):
        """Handle trump selection phase"""
        state = self.game.state
        
        if state.phase == GamePhase.TRUMP_SELECTION_ROUND1:
            card_display = self.format_card(state.turned_up_card)
            print(f"\n{Fore.YELLOW}Trump Selection - Round 1{Style.RESET_ALL}")
            print(f"Pick up the {card_display}?")
            print("  [1] Call it (pick up)")
            print("  [2] Pass")
            
            choice = input("\nYour choice: ").strip()
            
            if choice == '1':
                alone = input("Go alone? (y/n): ").strip().lower() == 'y'
                self.game.call_trump(go_alone=alone)
            else:
                self.game.pass_trump()
        
        elif state.phase == GamePhase.TRUMP_SELECTION_ROUND2:
            print(f"\n{Fore.YELLOW}Trump Selection - Round 2{Style.RESET_ALL}")
            print(f"Call a suit (not {state.turned_up_card.suit.value}):")
            print(f"  [C] {SUITS_DISPLAY['C']} Clubs")
            print(f"  [D] {SUITS_DISPLAY['D']} Diamonds")
            print(f"  [H] {SUITS_DISPLAY['H']} Hearts")
            print(f"  [S] {SUITS_DISPLAY['S']} Spades")
            if state.current_player_position != state.dealer_position:
                print("  [P] Pass")
            
            choice = input("\nYour choice: ").strip().upper()
            
            if choice == 'P' and state.current_player_position != state.dealer_position:
                self.game.pass_trump()
            elif choice in ['C', 'D', 'H', 'S']:
                try:
                    suit = Suit.from_string(choice)
                    if suit == state.turned_up_card.suit:
                        print(f"{Fore.RED}Cannot call the turned up suit!{Style.RESET_ALL}")
                        input("Press Enter to continue...")
                        return
                    alone = input("Go alone? (y/n): ").strip().lower() == 'y'
                    self.game.call_trump(suit, go_alone=alone)
                except Exception as e:
                    print(f"{Fore.RED}Invalid suit: {e}{Style.RESET_ALL}")
                    input("Press Enter to continue...")
    
    def handle_play_card(self, player_position: int):
        """Handle playing a card"""
        player = self.game.state.players[player_position]
        
        print(f"\n{Fore.YELLOW}Select a card to play:{Style.RESET_ALL}")
        
        while True:
            choice = input("Card number (0-4): ").strip()
            
            try:
                idx = int(choice)
                if 0 <= idx < len(player.hand):
                    card = player.hand[idx]
                    result = self.game.play_card(card)
                    
                    card_display = self.format_card(card)
                    print(f"\n{Fore.GREEN}You played: {card_display}{Style.RESET_ALL}")
                    
                    if result['trick_complete']:
                        winner = result['trick_winner']
                        print(f"{Fore.CYAN}Trick won by Player {winner}!{Style.RESET_ALL}")
                        
                        if result['hand_complete']:
                            hand_result = result['hand_winner']
                            print(f"\n{Fore.YELLOW}{'='*50}{Style.RESET_ALL}")
                            print(f"{Fore.GREEN}Hand Complete!{Style.RESET_ALL}")
                            print(f"Winning Team: {hand_result['winning_team']}")
                            print(f"Points Awarded: {hand_result['points_awarded']}")
                            print(f"Team 1 Score: {hand_result['team1_score']}")
                            print(f"Team 2 Score: {hand_result['team2_score']}")
                            print(f"{Fore.YELLOW}{'='*50}{Style.RESET_ALL}")
                            
                            if hand_result['game_over']:
                                print(f"\n{Fore.MAGENTA}GAME OVER!{Style.RESET_ALL}")
                                print(f"{Fore.GREEN}Team {hand_result['winning_team']} wins!{Style.RESET_ALL}")
                                input("\nPress Enter to exit...")
                                return
                        
                        input("\nPress Enter to continue...")
                    break
            except ValueError:
                print(f"{Fore.RED}Invalid input. Enter a number 0-4.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                print("Please choose a different card.")
    
    def run(self):
        """Main game loop"""
        self.setup_game()
        
        while self.game.state.phase != GamePhase.GAME_OVER:
            player_position = 0  # Human is always player 0
            
            self.display_game_state(player_position)
            
            # Handle different game phases
            if self.game.state.phase in [GamePhase.TRUMP_SELECTION_ROUND1, GamePhase.TRUMP_SELECTION_ROUND2]:
                if self.game.state.current_player_position == player_position:
                    self.handle_trump_selection(player_position)
                else:
                    # AI auto-pass for now (simplified)
                    input(f"AI Player {self.game.state.current_player_position} is deciding... (Press Enter)")
                    try:
                        self.game.pass_trump()
                    except:
                        # If dealer can't pass, call a random suit
                        available_suits = [s for s in Suit if s != self.game.state.turned_up_card.suit]
                        import random
                        self.game.call_trump(random.choice(available_suits))
            
            elif self.game.state.phase == GamePhase.PLAYING:
                if self.game.state.current_player_position == player_position:
                    self.handle_play_card(player_position)
                else:
                    # AI plays random valid card
                    import random
                    ai_player = self.game.state.get_current_player()
                    valid_cards = self.game.get_valid_moves()
                    if valid_cards:
                        card = random.choice(valid_cards)
                        input(f"AI Player {self.game.state.current_player_position} is playing... (Press Enter)")
                        self.game.play_card(card)
            
            elif self.game.state.phase == GamePhase.HAND_COMPLETE:
                # Start new hand
                input("Starting new hand... (Press Enter)")
                self.game.start_new_hand()


def main():
    cli = EuchreCLI()
    try:
        cli.run()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Game interrupted. Thanks for playing!{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
