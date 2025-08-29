// Initialize Socket.IO
const socket = io();

// DOM Elements
const analysisForm = document.getElementById('analysisForm');
const symbolsInput = document.getElementById('symbols');
const analysisResults = document.getElementById('analysisResults');
const portfolioPositions = document.getElementById('portfolioPositions');
const tradeHistory = document.getElementById('tradeHistory');

// Socket.IO Event Handlers
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('portfolio_update', (data) => {
    updatePortfolio(data.positions);
    updateTradeHistory(data.trade_history);
});

// Form Submit Handler
analysisForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const symbols = symbolsInput.value.split(',').map(s => s.trim()).filter(s => s);
    
    if (symbols.length === 0) {
        alert('Please enter at least one stock symbol');
        return;
    }

    // Show loading state
    analysisResults.innerHTML = '<div class="text-center"><div class="loading"></div> Analyzing...</div>';

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symbols })
        });

        const data = await response.json();
        
        if (response.ok) {
            displayAnalysisResults(data);
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
    } catch (error) {
        analysisResults.innerHTML = `<div class="alert alert-danger">${error.message}</div>`;
    }
});

// Helper Functions
function displayAnalysisResults(data) {
    let html = '';

    for (const symbol of data.symbols) {
        const technicalAnalysis = data.technical_analysis?.[symbol] || {};
        const fundamentals = data.fundamentals?.[symbol] || {};
        const sentiment = data.sentiment?.[symbol] || {};
        const md = data.market_data?.real_time?.[symbol] || null;

        // Market
        const priceStr = md && typeof md.price === 'number' ? `$${md.price.toFixed(2)}` : 'N/A';
        const tsStr = md?.timestamp ? new Date(md.timestamp).toLocaleString() : '—';

        // Technical
        const taClass = technicalAnalysis.overall_signal?.classification || 'N/A';
        const taStrength = technicalAnalysis.overall_signal?.strength;
        const taStrengthPct = (typeof taStrength === 'number') ? `${(taStrength * 100).toFixed(1)}%` : 'N/A';
        const taStrengthClass = getSignalClass(taStrength || 0);
        const bullishCount = technicalAnalysis.bullish_count ?? technicalAnalysis.bullish ?? '—';
        const bearishCount = technicalAnalysis.bearish_count ?? technicalAnalysis.bearish ?? '—';

        // Sentiment
        const sentClass = sentiment.sentiment_classification || 'N/A';
        const sentScore = (typeof sentiment.overall_sentiment === 'number') ? `${(sentiment.overall_sentiment * 100).toFixed(1)}%` : 'N/A';
        const sentConf = (typeof sentiment.confidence === 'number') ? `${(sentiment.confidence * 100).toFixed(0)}%` : 'N/A';
        const totalArticles = sentiment.total_articles || 0;
        const newsCount = sentiment.sources?.news?.article_count || 0;
        const socialCount = sentiment.sources?.social?.article_count || 0;
        const topNews = (sentiment.sources?.news?.articles || []).slice(0, 3);

        html += `
            <div class="analysis-result">
                <h4 class="mb-2">${symbol}</h4>

                <div class="row g-3">
                    <div class="col-md-4">
                        <div class="card h-100">
                            <div class="card-body">
                                <div class="d-flex justify-content-between align-items-center mb-2">
                                    <strong>Market</strong>
                                    <span class="badge bg-secondary">Real-time</span>
                                </div>
                                <div>
                                    <div>Price: <strong>${priceStr}</strong></div>
                                    <div class="text-muted"><small>As of ${tsStr}</small></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="col-md-4">
                        <div class="card h-100">
                            <div class="card-body">
                                <div class="d-flex justify-content-between align-items-center mb-2">
                                    <strong>Technical</strong>
                                    <span class="badge ${taStrengthClass}">${taStrengthPct}</span>
                                </div>
                                <div>
                                    <div>Signal: <strong>${taClass}</strong></div>
                                    <div class="text-muted"><small>Bullish: ${bullishCount} · Bearish: ${bearishCount}</small></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="col-md-4">
                        <div class="card h-100">
                            <div class="card-body">
                                <div class="d-flex justify-content-between align-items-center mb-2">
                                    <strong>Sentiment</strong>
                                    <span class="badge bg-info text-dark">Confidence ${sentConf}</span>
                                </div>
                                <div>
                                    <div>Overall: <strong>${sentClass}</strong> (${sentScore})</div>
                                    <div class="text-muted"><small>Total articles: ${totalArticles} · News: ${newsCount} · Social: ${socialCount}</small></div>
                                    <div class="mt-2">
                                        <small class="text-muted">Top news:</small>
                                        <ul class="mb-0">
                                            ${topNews.length ? topNews.map(a => `<li>${linkOrText(a)}</li>`).join('') : '<li class="text-muted">No recent news</li>'}
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="mt-2">
                    <strong>Fundamentals:</strong> ${fundamentals.recommendation || 'N/A'}
                </div>
            </div>
        `;
    }

    analysisResults.innerHTML = html;
}

function linkOrText(article) {
    const title = escapeHtml(article?.title || 'Untitled');
    const source = escapeHtml(article?.source || '');
    const url = article?.url;
    if (url) {
        return `<a href="${encodeURI(url)}" target="_blank" rel="noopener noreferrer">${title}</a> <small class="text-muted">${source}</small>`;
    }
    return `${title} <small class="text-muted">${source}</small>`;
}

function escapeHtml(text) {
    if (!text) return '';
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function updatePortfolio(positions) {
    let html = '';
    
    if (positions.positions && positions.positions.length > 0) {
        for (const position of positions.positions) {
            const profitClass = position.unrealized_pnl >= 0 ? 'profit' : 'loss';
            html += `
                <div class="position-item ${profitClass}">
                    <h5>${position.symbol}</h5>
                    <div>Shares: ${position.size}</div>
                    <div>Entry Price: $${position.entry_price.toFixed(2)}</div>
                    <div>Current Price: $${position.current_price.toFixed(2)}</div>
                    <div>P/L: $${position.unrealized_pnl.toFixed(2)} (${(position.unrealized_pnl_pct * 100).toFixed(2)}%)</div>
                </div>
            `;
        }
    } else {
        html = '<div class="alert alert-info">No open positions</div>';
    }

    portfolioPositions.innerHTML = html;
}

function updateTradeHistory(trades) {
    let html = '';
    
    if (trades.length > 0) {
        trades.reverse().forEach(trade => {
            const tradeClass = trade.size > 0 ? 'buy' : 'sell';
            html += `
                <div class="trade-item ${tradeClass}">
                    <div class="d-flex justify-content-between">
                        <strong>${trade.symbol}</strong>
                        <small>${new Date(trade.timestamp).toLocaleString()}</small>
                    </div>
                    <div>
                        ${trade.size > 0 ? 'Buy' : 'Sell'} ${Math.abs(trade.size)} shares @ $${trade.price.toFixed(2)}
                    </div>
                </div>
            `;
        });
    } else {
        html = '<div class="alert alert-info">No trades yet</div>';
    }

    tradeHistory.innerHTML = html;
}

function getSignalClass(strength) {
    if (strength >= 0.4) return 'signal-strong';
    if (strength >= 0.25) return 'signal-moderate';
    return 'signal-weak';
}

// Initial load of portfolio and trade history
async function loadInitialData() {
    try {
        const response = await fetch('/api/portfolio');
        const data = await response.json();
        
        if (response.ok) {
            updatePortfolio(data.positions);
            updateTradeHistory(data.trade_history);
        }
    } catch (error) {
        console.error('Failed to load initial data:', error);
    }
}

loadInitialData();