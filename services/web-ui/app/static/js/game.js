/**
 * Game UI JavaScript - Handles card highlighting based on valid moves
 */

// Configuration
const POLL_INTERVAL = 2000; // Poll every 2 seconds
let pollTimer = null;
let currentGameId = null;
let currentPlayerPosition = null;
let validCards = [];

/**
 * Initialize the game UI
 */
function initializeGame(gameId, playerPosition) {
    currentGameId = gameId;
    currentPlayerPosition = playerPosition;
    
    // Start polling for valid moves
    startPolling();
    
    // Initial fetch
    fetchValidMoves();
}

/**
 * Start polling for valid moves
 */
function startPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
    }
    
    pollTimer = setInterval(fetchValidMoves, POLL_INTERVAL);
}

/**
 * Stop polling
 */
function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

/**
 * Fetch valid moves from the API
 */
async function fetchValidMoves() {
    if (!currentGameId) {
        return;
    }
    
    try {
        const response = await fetch(`/api/games/${currentGameId}/valid-moves?perspective=${currentPlayerPosition}`);
        
        if (!response.ok) {
            console.error('Failed to fetch valid moves:', response.statusText);
            return;
        }
        
        const data = await response.json();
        validCards = data.valid_cards || [];
        
        // Update card highlighting
        updateCardHighlighting(data);
        
    } catch (error) {
        console.error('Error fetching valid moves:', error);
    }
}

/**
 * Update card highlighting based on valid moves
 */
function updateCardHighlighting(validMovesData) {
    const cardButtons = document.querySelectorAll('button[name="card"]');
    
    cardButtons.forEach(button => {
        const cardValue = button.value;
        const isValid = validMovesData.valid_cards.includes(cardValue);
        
        // Remove existing classes
        button.classList.remove('valid-card', 'invalid-card');
        
        // Add appropriate class
        if (isValid) {
            button.classList.add('valid-card');
            button.disabled = false;
            button.style.cursor = 'pointer';
        } else {
            button.classList.add('invalid-card');
            button.disabled = true;
            button.style.cursor = 'not-allowed';
        }
    });
    
    // Update info display if needed
    updateGameInfo(validMovesData);
}

/**
 * Update game info display
 */
function updateGameInfo(validMovesData) {
    const infoElement = document.getElementById('valid-moves-info');
    if (!infoElement) {
        return;
    }
    
    let infoText = '';
    
    if (validMovesData.must_follow_suit && validMovesData.current_trick_lead_suit) {
        const suitNames = {
            'H': 'Hearts ♥',
            'D': 'Diamonds ♦',
            'C': 'Clubs ♣',
            'S': 'Spades ♠'
        };
        const suitName = suitNames[validMovesData.current_trick_lead_suit] || validMovesData.current_trick_lead_suit;
        infoText = `You must follow suit: ${suitName}`;
    } else if (validMovesData.valid_cards.length > 0) {
        infoText = `You can play any of ${validMovesData.valid_cards.length} cards`;
    }
    
    infoElement.textContent = infoText;
}

/**
 * Check if a card is valid to play
 */
function isCardValid(cardValue) {
    return validCards.includes(cardValue);
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopPolling();
});
