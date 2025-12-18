"""
Main web UI routes
"""

from flask import Blueprint, render_template, session, request, redirect, url_for, current_app
import requests

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@main_bp.route('/new-game', methods=['GET', 'POST'])
def new_game():
    """Create a new game"""
    if request.method == 'POST':
        # Get player names from form
        players = [
            {'name': request.form.get('player0', 'Player 1'), 'type': 'human'},
            {'name': request.form.get('player1', 'Player 2'), 'type': request.form.get('type1', 'random_ai')},
            {'name': request.form.get('player2', 'Player 3'), 'type': request.form.get('type2', 'random_ai')},
            {'name': request.form.get('player3', 'Player 4'), 'type': request.form.get('type3', 'random_ai')},
        ]
        
        # Create game via API
        api_url = current_app.config['EUCHRE_API_URL']
        try:
            response = requests.post(
                f"{api_url}/api/games",
                json={'players': players},
                timeout=5
            )
            
            if response.status_code == 201:
                data = response.json()
                session['game_id'] = data['game_id']
                session['player_position'] = 0
                return redirect(url_for('main.play_game'))
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    return render_template('new_game.html')


@main_bp.route('/game')
def play_game():
    """Play game interface"""
    game_id = session.get('game_id')
    if not game_id:
        return redirect(url_for('main.index'))
    
    player_position = session.get('player_position', 0)
    
    # Get game state from API
    api_url = current_app.config['EUCHRE_API_URL']
    try:
        response = requests.get(
            f"{api_url}/api/games/{game_id}",
            params={'perspective': player_position},
            timeout=5
        )
        
        if response.status_code == 200:
            game_state = response.json()
            return render_template('game.html', game_state=game_state, player_position=player_position)
    except Exception as e:
        return render_template('error.html', error=str(e))
    
    return render_template('game.html', game_state={}, player_position=player_position)


@main_bp.route('/history')
def history():
    """View game history"""
    return render_template('history.html')


@main_bp.route('/training')
def training():
    """AI training dashboard"""
    return render_template('training.html')
