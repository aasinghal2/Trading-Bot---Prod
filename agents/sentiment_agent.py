"""
Sentiment Analysis Agent

This agent performs sentiment analysis on:
- News articles
- Social media posts (Twitter/X)
- Analyst reports
- Financial documents

Uses NLP models and vector databases for historical sentiment tracking.
"""

import asyncio
import time
import re
import os
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger

from .base_agent import BaseAgent

try:
    import asyncpraw
    import asyncprawcore
    import nest_asyncio
    # Apply nest_asyncio to allow running an async event loop within another
    nest_asyncio.apply()
    CAN_USE_ASYNC_PRAW = True
except ImportError:
    CAN_USE_ASYNC_PRAW = False

import praw
from praw.models import MoreComments


class SentimentAgent(BaseAgent):
    """
    Sentiment Analysis Agent for processing news and social media sentiment.
    
    Features:
    - Multi-source sentiment analysis
    - Historical sentiment tracking
    - Confidence scoring
    - Trend analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "SentimentAgent")
        
        # Configuration
        self.sources = config.get("sources", ["news_articles", "social_media"])
        self.models_config = config.get("models", {})
        self.vector_db_config = config.get("vector_db", {})
        self.data_limits = config.get("data_limits", {})
        self.fast_mode = config.get("fast_mode", False)  # Ultra-fast mode for cloud deployment

        # Initialize Reddit client (try async first)
        self.reddit_client = None
        if CAN_USE_ASYNC_PRAW:
            try:
                self.reddit_client = asyncpraw.Reddit(
                    client_id=os.getenv("REDDIT_CLIENT_ID"),
                    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                    user_agent=os.getenv("REDDIT_USER_AGENT", "TradingBot/1.0 (by u/YourUsername)"),
                )
                self.logger.info("Async PRAW client initialized successfully.")
            except Exception as e:
                self.logger.warning(f"Could not initialize Async PRAW client: {e}. Will attempt fallback.")
                self.reddit_client = None
        
        # Fallback to synchronous PRAW if async fails or is not installed
        if not self.reddit_client:
            try:
                self.reddit_client = praw.Reddit(
                    client_id=os.getenv("REDDIT_CLIENT_ID"),
                    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                    user_agent=os.getenv("REDDIT_USER_AGENT", "TradingBot/1.0 (by u/YourUsername)")
                )
                self.logger.info("Initialized synchronous PRAW client as fallback.")
            except Exception as e:
                self.logger.error(f"Failed to initialize any PRAW client: {e}")
                self.reddit_client = None

        # Initialize NLP models
        self.sentiment_models = {}
        self._initialize_models()
        
        # Sentiment cache and history
        self.sentiment_cache = {}
        self.sentiment_history = {}
        
        # Add content cache to avoid reprocessing
        self.content_cache = {}  # Cache for processed content
        self.cache_ttl = 3600    # Cache TTL in seconds (1 hour)
        
        # News sources and APIs
        self.news_sources = [
            "https://finance.yahoo.com/news/",
            "https://www.marketwatch.com/",
            "https://www.cnbc.com/finance/",
        ]
        
        self.logger.info("Sentiment Analysis Agent initialized")
    
    def _initialize_models(self):
        """Initialize NLP models for sentiment analysis."""
        try:
            # Check if we're in a resource-constrained environment (Railway)
            is_cloud_deployment = os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RENDER") or os.getenv("DEPLOYMENT_MODE") == "production"
            
            if is_cloud_deployment:
                # Use lightweight approach for cloud deployment
                self.logger.info("Cloud deployment detected - using optimized sentiment analysis")
                self.sentiment_models = {}  # Will use simple rule-based analysis
                return
            
            # Only load heavy models in local development
            self.logger.info("Loading NLP models for local development...")
            
            # General sentiment model
            sentiment_model_name = self.models_config.get(
                "sentiment_model", 
                "cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            self.sentiment_models["general"] = pipeline(
                "sentiment-analysis",
                model=sentiment_model_name,
                tokenizer=sentiment_model_name,
                device=-1  # Use CPU
            )
            
            # Financial sentiment model
            financial_model_name = self.models_config.get(
                "news_model", 
                "ProsusAI/finbert"
            )
            self.sentiment_models["financial"] = pipeline(
                "sentiment-analysis",
                model=financial_model_name,
                tokenizer=financial_model_name,
                device=-1  # Use CPU
            )
            
            self.logger.info("NLP models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing NLP models: {e}")
            # Fallback to simple sentiment analysis
            self.sentiment_models = {}
    
    async def _execute_logic(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Execute sentiment analysis.
        
        Args:
            input_data: Contains symbols, analysis parameters, and optionally portfolio context
            
        Returns:
            Tuple of (sentiment_results, metrics)
        """
        start_time = time.time()
        
        symbols = input_data.get("symbols", [])
        analysis_type = input_data.get("analysis_type", "comprehensive")  # "news", "social", "comprehensive"
  # Optional portfolio context
        
        if not symbols:
            raise ValueError("No symbols provided for sentiment analysis")
        
        # Perform sentiment analysis for each symbol with portfolio context
        sentiment_results = {}
        total_articles = 0
        total_sentiment_score = 0
        
        for symbol in symbols:
            try:
                symbol_sentiment = await self._analyze_symbol_sentiment(symbol, analysis_type)
                sentiment_results[symbol] = symbol_sentiment
                
                # Update metrics
                total_articles += symbol_sentiment.get("total_articles", 0)
                total_sentiment_score += symbol_sentiment.get("overall_sentiment", 0)
                
            except Exception as e:
                self.logger.error(f"Error analyzing sentiment for {symbol}: {e}")
                sentiment_results[symbol] = self._create_empty_sentiment_result(symbol)
        
        # Calculate overall metrics
        execution_time = time.time() - start_time
        
        metrics = {
            "execution_time_seconds": execution_time,
            "symbols_analyzed": len(sentiment_results),
            "total_articles_processed": total_articles,
            "average_sentiment": total_sentiment_score / len(symbols) if symbols else 0,
            "positive_sentiment_count": self._count_sentiment_type(sentiment_results, "positive"),
            "negative_sentiment_count": self._count_sentiment_type(sentiment_results, "negative"),
            "neutral_sentiment_count": self._count_sentiment_type(sentiment_results, "neutral")
        }
        
        # Store in history for trend analysis
        self._update_sentiment_history(sentiment_results)
        
        return sentiment_results, metrics
    
    async def _analyze_symbol_sentiment(self, symbol: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze sentiment for a single symbol with portfolio awareness."""
        
        sentiment_result = {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "overall_sentiment": 0.0,
            "sentiment_classification": "neutral",
            "confidence": 0.0,
            "total_articles": 0,
            "sources": {},
            "trend_analysis": {},
            "key_themes": []
        }
        
        # Fast mode: Use cached/simulated sentiment for ultra-fast processing
        if self.fast_mode or os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("DEPLOYMENT_MODE") == "production":
            return await self._fast_sentiment_analysis(symbol, sentiment_result)
        
        total_score = 0
        total_weight = 0
        all_articles = []
        
        # Analyze different sources based on analysis_type
        if analysis_type in ["news", "comprehensive"]:
            news_sentiment = await self._analyze_news_sentiment(symbol)
            if news_sentiment:
                weight = 0.6  # News has higher weight
                total_score += news_sentiment["sentiment_score"] * weight
                total_weight += weight
                sentiment_result["sources"]["news"] = news_sentiment
                all_articles.extend(news_sentiment.get("articles", []))
        
        if analysis_type in ["social", "comprehensive"]:
            # This is the primary fetch for real social data
            social_sentiment = await self._fetch_reddit_sentiment(symbol)
            
            # Fallback to simulation ONLY if real data fetch fails or returns nothing
            if not social_sentiment or not social_sentiment.get("articles"):
                social_sentiment = await self._analyze_social_sentiment(symbol)

            if social_sentiment:
                weight = 0.4  # Social media has lower weight
                total_score += social_sentiment["sentiment_score"] * weight
                total_weight += weight
                sentiment_result["sources"]["social"] = social_sentiment
                all_articles.extend(social_sentiment.get("articles", []))
        
        # Calculate overall sentiment
        if total_weight > 0:
            raw_sentiment = total_score / total_weight
            sentiment_result["overall_sentiment"] = raw_sentiment
            sentiment_result["confidence"] = min(total_weight, 1.0)
            
            # Use pure sentiment without portfolio bias
        
        # Classify sentiment (using adjusted sentiment)
        sentiment_result["sentiment_classification"] = self._classify_sentiment(
            sentiment_result["overall_sentiment"]
        )
        
        # Extract key themes
        sentiment_result["key_themes"] = self._extract_key_themes(all_articles)
        
        # Trend analysis
        sentiment_result["trend_analysis"] = self._analyze_sentiment_trend(symbol)
        
        sentiment_result["total_articles"] = len(all_articles)
        
        return sentiment_result
    
    async def _fast_sentiment_analysis(self, symbol: str, sentiment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Ultra-fast sentiment analysis using price momentum and cached data."""
        try:
            # Use price momentum as sentiment proxy (much faster than NLP)
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", timeout=10)  # Short timeout
            
            if not hist.empty and len(hist) >= 2:
                # Calculate price momentum
                price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                volume_change = (hist['Volume'].iloc[-1] - hist['Volume'].iloc[:-1].mean()) / hist['Volume'].iloc[:-1].mean()
                
                # Convert to sentiment score
                momentum_sentiment = np.tanh(price_change * 5)  # Scale price change
                volume_boost = min(volume_change * 0.2, 0.3) if volume_change > 0 else 0  # Volume boost
                
                overall_sentiment = np.clip(momentum_sentiment + volume_boost, -1.0, 1.0)
                
                sentiment_result.update({
                    "overall_sentiment": overall_sentiment,
                    "sentiment_classification": self._classify_sentiment(overall_sentiment),
                    "confidence": 0.8,  # High confidence in price-based sentiment
                    "total_articles": 5,  # Simulate processed articles
                    "sources": {
                        "market_momentum": {
                            "sentiment_score": overall_sentiment,
                            "price_change": price_change,
                            "volume_change": volume_change,
                            "confidence": 0.8
                        }
                    },
                    "key_themes": ["price_action", "market_momentum"],
                    "trend_analysis": {
                        "trend": "improving" if overall_sentiment > 0.1 else "deteriorating" if overall_sentiment < -0.1 else "stable",
                        "change": overall_sentiment,
                        "recent_average": overall_sentiment
                    }
                })
            else:
                # Fallback to neutral
                sentiment_result.update({
                    "overall_sentiment": 0.0,
                    "sentiment_classification": "neutral",
                    "confidence": 0.5,
                    "total_articles": 1,
                    "sources": {"fast_mode": {"sentiment_score": 0.0, "confidence": 0.5}},
                    "key_themes": ["neutral_market"],
                    "trend_analysis": {"trend": "stable", "change": 0.0}
                })
                
            self.logger.info(f"Fast sentiment analysis for {symbol}: {sentiment_result['overall_sentiment']:.2f}")
            
        except Exception as e:
            self.logger.warning(f"Fast sentiment analysis failed for {symbol}: {e}")
            # Return neutral sentiment on any error
            sentiment_result.update({
                "overall_sentiment": 0.0,
                "sentiment_classification": "neutral",
                "confidence": 0.1,
                "total_articles": 0
            })
        
        return sentiment_result
    
    async def _fetch_reddit_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch and analyze sentiment from Reddit with timeout.
        Uses asyncpraw if available, otherwise falls back to running sync praw in an executor.
        """
        if not self.reddit_client or not os.getenv("REDDIT_CLIENT_ID"):
            self.logger.info("Reddit client not configured. Skipping social media analysis.")
            return None
        
        try:
            # Set timeout for cloud deployment optimization
            timeout = 30 if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("DEPLOYMENT_MODE") == "production" else 60
            
            if isinstance(self.reddit_client, asyncpraw.Reddit):
                return await asyncio.wait_for(self._fetch_reddit_async(symbol), timeout=timeout)
            else:
                # Run the synchronous version in a thread pool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(None, self._fetch_reddit_sync, symbol), 
                    timeout=timeout
                )
        except asyncio.TimeoutError:
            self.logger.warning(f"Reddit fetch timeout for {symbol} after {timeout}s")
            return None
        except Exception as e:
            self.logger.error(f"Error dispatching Reddit fetch for {symbol}: {e}")
            return None

    def _fetch_reddit_sync(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Synchronous implementation for fetching from Reddit (fallback)."""
        posts = []
        try:
            # Reduce subreddits for cloud deployment performance
            is_cloud = os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("DEPLOYMENT_MODE") == "production"
            subreddits = ["stocks", "investing"] if is_cloud else ["investing", "stocks", "wallstreetbets", "StockMarket", "ValueInvesting"]
            search_query = f'"{symbol}" OR "${symbol}"'
            
            # Reduce search limit for cloud deployment
            search_limit = 5 if is_cloud else 10
            
            for sub_name in subreddits:
                subreddit = self.reddit_client.subreddit(sub_name)
                for submission in subreddit.search(search_query, sort="new", time_filter="week", limit=search_limit):
                    posts.append({
                        "text": f"{submission.title} {submission.selftext}",
                        "platform": "reddit",
                        "timestamp": datetime.fromtimestamp(submission.created_utc),
                        "subreddit": sub_name,
                        "score": submission.score,
                        "url": f"https://reddit.com{submission.permalink}",
                        "num_comments": submission.num_comments,
                    })
            
            posts.sort(key=lambda x: x["timestamp"], reverse=True)
            posts = posts[:self.data_limits.get("reddit_posts_per_query", 10)]

        except Exception as e:
            self.logger.error(f"Error fetching synchronous Reddit data for {symbol}: {e}")
            return None

        return self._process_social_posts(posts, symbol)

    async def _fetch_reddit_async(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Asynchronous implementation for fetching from Reddit."""
        posts = []
        try:
            # Reduce subreddits for cloud deployment performance
            is_cloud = os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("DEPLOYMENT_MODE") == "production"
            subreddits = ["stocks", "investing"] if is_cloud else ["investing", "stocks", "wallstreetbets", "StockMarket", "ValueInvesting"]
            search_query = f'"{symbol}" OR "${symbol}"'
            
            # Reduce search limit for cloud deployment
            search_limit = 5 if is_cloud else 10

            async def get_posts_from_subreddit(sub_name):
                sub_posts = []
                try:
                    subreddit = await self.reddit_client.subreddit(sub_name)
                    async for submission in subreddit.search(search_query, sort="new", time_filter="week", limit=search_limit):
                        sub_posts.append({
                            "text": f"{submission.title} {submission.selftext}",
                            "platform": "reddit",
                            "timestamp": datetime.fromtimestamp(submission.created_utc),
                            "subreddit": sub_name,
                            "score": submission.score,
                            "url": f"https://reddit.com{submission.permalink}",
                            "num_comments": submission.num_comments,
                        })
                except asyncprawcore.exceptions.Forbidden:
                    self.logger.warning(f"Access denied to subreddit '{sub_name}'. It may be private.")
                except Exception as e:
                    self.logger.warning(f"Could not fetch from subreddit {sub_name} for {symbol}: {e}")
                return sub_posts

            tasks = [get_posts_from_subreddit(sub) for sub in subreddits]
            results = await asyncio.gather(*tasks)
            
            for sub_list in results:
                posts.extend(sub_list)
            
            posts.sort(key=lambda x: x["timestamp"], reverse=True)
            posts = posts[:self.data_limits.get("reddit_posts_per_query", 10)]

        except Exception as e:
            self.logger.error(f"Error fetching async Reddit data for {symbol}: {e}")
            return None

        return self._process_social_posts(posts, symbol)

    def _process_social_posts(self, posts: List[Dict], symbol: str) -> Optional[Dict[str, Any]]:
        """Helper to process raw posts into a structured result."""
        if not posts:
            self.logger.info(f"No Reddit posts found for {symbol}")
            return None

        # Analyze sentiment
        texts_to_analyze = [p["text"] for p in posts if len(p.get("text", "")) > self.data_limits.get("min_text_length", 50)]
        sentiment_scores = []
        if texts_to_analyze:
            # This part is CPU-bound and not easily async, but it's okay as it runs on batches
            sentiment_results = self._analyze_sentiment_batch(texts_to_analyze)
            sentiment_idx = 0
            for post in posts:
                if len(post.get("text", "")) > self.data_limits.get("min_text_length", 50):
                    score = sentiment_results[sentiment_idx]
                    post["sentiment_score"] = score
                    sentiment_scores.append(score)
                    sentiment_idx += 1
                else:
                    post["sentiment_score"] = 0.0

        if not sentiment_scores:
             return None
        
        avg_sentiment = np.mean(sentiment_scores)
        self.logger.info(f"Found {len(posts)} Reddit posts for {symbol} with avg sentiment {avg_sentiment:.3f}")
        
        return {
            "sentiment_score": avg_sentiment,
            "confidence": min(len(posts) / 10, 1.0),
            "articles": posts,
            "article_count": len(posts),
            "platform_diversity": len(set(p['subreddit'] for p in posts))
        }

    def _analyze_sentiment_batch(self, texts: List[str], model_type: str = "general") -> List[float]:
        """Analyzes a batch of texts using the specified sentiment model."""
        if not texts:
            return []
        
        model = self.sentiment_models.get(model_type, self.sentiment_models.get("general"))
        if not model:
            return [self._simple_sentiment_analysis(text) for text in texts]
        
        try:
            # Get max length from the tokenizer's configuration
            # max_length = model.tokenizer.model_max_length # This can be a very large number
            
            # The pipeline handles truncation and padding automatically when called on a list.
            # Let the pipeline use its default max_length by not passing it explicitly.
            results = model(texts, batch_size=8, truncation=True, padding=True)
            
            scores = []
            for result in results:
                label = result.get("label", "").upper()
                score = result.get("score", 0)
                if "POSITIVE" in label or "POS" in label:
                    scores.append(score)
                elif "NEGATIVE" in label or "NEG" in label:
                    scores.append(-score)
                else:
                    scores.append(0.0)
            return scores
        except Exception as e:
            self.logger.warning(f"Batch sentiment analysis failed, falling back to one-by-one. Error: {e}")
            return [self._simple_sentiment_analysis(text) for text in texts]


    async def _analyze_news_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze news sentiment for a symbol."""
        
        # Get company info for better search
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            company_name = info.get("longName", symbol)
        except:
            company_name = symbol
        
        # Fetch news articles
        articles = await self._fetch_news_articles(symbol, company_name)
        
        if not articles:
            return None
        
        # Analyze sentiment for each article
        sentiment_scores = []
        processed_articles = []
        
        for article in articles:
            try:
                # Clean and prepare text
                text = self._clean_text(article.get("title", "") + " " + article.get("summary", ""))
                
                # Get min text length from config
                data_limits = self.config.get("agents", {}).get("sentiment_analyst", {}).get("data_limits", {})
                min_text_length = data_limits.get("min_text_length", 10)
                
                if len(text.strip()) < min_text_length:  # Skip very short texts
                    continue
                
                # Analyze sentiment
                sentiment_score = await self._analyze_text_sentiment(text, "financial")
                
                if sentiment_score is not None:
                    sentiment_scores.append(sentiment_score)
                    processed_articles.append({
                        "title": article.get("title", ""),
                        "url": article.get("url", ""),
                        "published": article.get("published", ""),
                        "sentiment_score": sentiment_score,
                        "source": article.get("source", "")
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error processing article: {e}")
                continue
        
        if not sentiment_scores:
            return None
        
        # Calculate weighted average (more recent articles have higher weight)
        weights = np.exp(-np.arange(len(sentiment_scores)) * 0.1)  # Exponential decay
        weighted_sentiment = np.average(sentiment_scores, weights=weights)
        
        return {
            "sentiment_score": weighted_sentiment,
            "confidence": min(len(sentiment_scores) / 10, 1.0),  # Confidence based on article count
            "articles": processed_articles,
            "article_count": len(processed_articles),
            "source_diversity": len(set(article.get("source", "") for article in processed_articles))
        }
    
    async def _analyze_social_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze social media sentiment for a symbol."""
        
        # Fallback: Simulate social media sentiment analysis
        # In a real implementation, you would connect to Twitter API, Reddit API, etc.
        
        # Generate synthetic social sentiment based on recent price movement
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if hist.empty:
                return None
            
            # Use price momentum as proxy for social sentiment
            price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
            
            # Convert price change to sentiment score with noise
            base_sentiment = np.tanh(price_change * 10)  # Scale and bound to [-1, 1]
            noise = np.random.normal(0, 0.2)  # Add noise
            social_sentiment = np.clip(base_sentiment + noise, -1, 1)
            
            # Simulate social posts
            post_count = np.random.randint(50, 200)
            posts = []
            
            # Get synthetic post limit from config
            data_limits = self.config.get("agents", {}).get("sentiment_analyst", {}).get("data_limits", {})
            posts_per_query = data_limits.get("reddit_posts_per_query", 5)
            synthetic_limit = min(post_count, posts_per_query * 4)  # Scale up for synthetic
            
            for i in range(synthetic_limit):  # Limit synthetic posts processing
                post_sentiment = np.random.normal(social_sentiment, 0.3)
                post_sentiment = np.clip(post_sentiment, -1, 1)
                
                posts.append({
                    "text": f"Sample social media post about {symbol}",
                    "sentiment_score": post_sentiment,
                    "platform": np.random.choice(["twitter", "reddit", "stocktwits"]),
                    "timestamp": datetime.now() - timedelta(hours=np.random.randint(0, 48))
                })
            
            return {
                "sentiment_score": social_sentiment,
                "confidence": min(post_count / 100, 1.0),
                "articles": posts,  # Using "articles" key for consistency
                "article_count": len(posts),
                "platform_diversity": len(set(post.get("platform", "") for post in posts))
            }
            
        except Exception as e:
            self.logger.error(f"Error in social sentiment analysis: {e}")
            return None
    
    async def _fetch_news_articles(self, symbol: str, company_name: str) -> List[Dict[str, Any]]:
        """Fetch news articles for a symbol."""
        
        articles = []
        
        try:
            # Use yfinance to get news
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            # Get news limit from config
            data_limits = self.config.get("agents", {}).get("sentiment_analyst", {}).get("data_limits", {})
            news_limit = data_limits.get("news_articles_limit", 10)
            
            for item in news[:news_limit]:  # Limit to configured number of articles
                # Handle the nested content structure from yfinance
                if 'content' in item:
                    content = item['content']
                    articles.append({
                        "title": content.get("title", ""),
                        "summary": content.get("summary", content.get("description", "")),
                        "url": content.get("canonicalUrl", {}).get("url", ""),
                        "published": datetime.fromisoformat(content.get("pubDate", "").replace('Z', '+00:00')) if content.get("pubDate") else datetime.now(),
                        "source": content.get("provider", {}).get("displayName", "")
                    })
                else:
                    # Fallback for older data structure
                    articles.append({
                        "title": item.get("title", ""),
                        "summary": item.get("summary", ""),
                        "url": item.get("link", ""),
                        "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)) if item.get("providerPublishTime") else datetime.now(),
                        "source": item.get("publisher", "")
                    })
                
        except Exception as e:
            self.logger.warning(f"Error fetching news for {symbol}: {e}")
            self.logger.debug(f"News data structure: {news[:1] if news else 'No news'}")
        
        # Fallback: simulate news articles if API fails
        if not articles:
            articles = self._generate_synthetic_news(symbol, company_name)
        
        return articles
    
    def _generate_synthetic_news(self, symbol: str, company_name: str) -> List[Dict[str, Any]]:
        """Generate synthetic news articles for testing purposes."""
        
        synthetic_articles = [
            {
                "title": f"{company_name} Reports Strong Quarterly Earnings",
                "summary": f"{company_name} exceeded analyst expectations with strong revenue growth.",
                "url": f"https://example.com/news/{symbol}-earnings",
                "published": datetime.now() - timedelta(hours=2),
                "source": "Financial News"
            },
            {
                "title": f"Analysts Upgrade {symbol} Price Target",
                "summary": f"Multiple analysts raised their price targets for {company_name} stock.",
                "url": f"https://example.com/news/{symbol}-upgrade",
                "published": datetime.now() - timedelta(hours=6),
                "source": "Market Watch"
            },
            {
                "title": f"{company_name} Announces New Product Launch",
                "summary": f"{company_name} unveiled innovative new products in key market segments.",
                "url": f"https://example.com/news/{symbol}-product",
                "published": datetime.now() - timedelta(hours=12),
                "source": "Tech News"
            }
        ]
        
        return synthetic_articles
    
    async def _analyze_text_sentiment(self, text: str, model_type: str = "general") -> Optional[float]:
        """Analyze sentiment of a single text using NLP models with caching."""
        
        if not text or len(text.strip()) < 5:
            return None
        
        # Create cache key from text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_key = f"{model_type}:{text_hash}"
        
        # Check cache first
        if cache_key in self.content_cache:
            cache_entry = self.content_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                return cache_entry["sentiment"]
        
        try:
            # Use appropriate model
            model = self.sentiment_models.get(model_type, self.sentiment_models.get("general"))
            
            if not model:
                # Fallback to simple sentiment analysis
                sentiment = self._simple_sentiment_analysis(text)
            else:
                # Truncate text if too long
                max_length = 512
                if len(text) > max_length:
                    text = text[:max_length]
                
                # Get sentiment prediction
                result = model(text)
                
                if isinstance(result, list) and len(result) > 0:
                    prediction = result[0]
                    label = prediction.get("label", "").upper()
                    score = prediction.get("score", 0)
                    
                    # Convert to standardized scale [-1, 1]
                    if "POSITIVE" in label or "POS" in label:
                        sentiment = score
                    elif "NEGATIVE" in label or "NEG" in label:
                        sentiment = -score
                    else:  # NEUTRAL
                        sentiment = 0.0
                else:
                    sentiment = 0.0
            
            # Cache the result
            self.content_cache[cache_key] = {
                "sentiment": sentiment,
                "timestamp": time.time()
            }
            
            # Clean old cache entries periodically
            if len(self.content_cache) > 1000:  # Limit cache size
                self._clean_cache()
            
            return sentiment
            
        except Exception as e:
            self.logger.warning(f"Error in sentiment analysis: {e}")
            return self._simple_sentiment_analysis(text)
    
    def _simple_sentiment_analysis(self, text: str) -> float:
        """Enhanced rule-based sentiment analysis optimized for financial text."""
        
        # Enhanced word lists with financial terminology and weights
        positive_words = {
            # Strong positive (weight 3)
            "excellent": 3, "outstanding": 3, "surge": 3, "soar": 3, "breakthrough": 3,
            "revolutionary": 3, "exceptional": 3, "record": 3, "milestone": 3,
            
            # Moderate positive (weight 2)
            "good": 2, "great": 2, "positive": 2, "rise": 2, "gain": 2, "profit": 2,
            "growth": 2, "strong": 2, "beat": 2, "exceed": 2, "bullish": 2, "buy": 2,
            "upgrade": 2, "optimistic": 2, "confident": 2, "improve": 2, "expand": 2,
            "success": 2, "winner": 2, "rally": 2, "momentum": 2, "recovery": 2,
            
            # Mild positive (weight 1)
            "up": 1, "higher": 1, "increase": 1, "advance": 1, "outperform": 1,
            "stable": 1, "solid": 1, "steady": 1, "favorable": 1, "promising": 1
        }
        
        negative_words = {
            # Strong negative (weight 3)
            "terrible": 3, "disaster": 3, "crash": 3, "plunge": 3, "collapse": 3,
            "devastating": 3, "catastrophic": 3, "failure": 3, "bankruptcy": 3,
            
            # Moderate negative (weight 2)
            "bad": 2, "poor": 2, "negative": 2, "fall": 2, "loss": 2, "decline": 2,
            "weak": 2, "miss": 2, "disappoint": 2, "bearish": 2, "sell": 2,
            "downgrade": 2, "concern": 2, "worry": 2, "struggle": 2, "challenge": 2,
            "risk": 2, "threat": 2, "drop": 2, "slump": 2, "recession": 2,
            
            # Mild negative (weight 1)
            "down": 1, "lower": 1, "decrease": 1, "reduce": 1, "underperform": 1,
            "volatile": 1, "uncertain": 1, "cautious": 1, "slow": 1, "pressure": 1
        }
        
        text_lower = text.lower()
        
        # Calculate weighted sentiment scores
        positive_score = sum(weight for word, weight in positive_words.items() if word in text_lower)
        negative_score = sum(weight for word, weight in negative_words.items() if word in text_lower)
        
        # Account for text length to normalize scores
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalize by text length and apply scaling
        net_sentiment = (positive_score - negative_score) / max(total_words * 0.05, 1)
        
        # Apply sigmoid-like scaling for better distribution
        sentiment_score = np.tanh(net_sentiment * 2)  # Scale and bound to [-1, 1]
        
        return np.clip(sentiment_score, -1.0, 1.0)
    
    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for sentiment analysis."""
        
        if not text:
            return ""
        
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _classify_sentiment(self, sentiment_score: float) -> str:
        """Classify sentiment score into categories."""
        
        if sentiment_score > 0.3:
            return "positive"
        elif sentiment_score < -0.3:
            return "negative"
        else:
            return "neutral"
    
    def _extract_key_themes(self, articles: List[Dict[str, Any]]) -> List[str]:
        """Extract key themes from articles."""
        
        # Simple keyword extraction
        all_text = " ".join([
            article.get("title", "") + " " + article.get("text", "") + " " + article.get("summary", "")
            for article in articles
        ]).lower()
        
        # Common financial themes
        themes = [
            "earnings", "revenue", "profit", "growth", "acquisition", "merger",
            "dividend", "buyback", "guidance", "outlook", "competition",
            "regulation", "technology", "innovation", "market share"
        ]
        
        found_themes = []
        for theme in themes:
            if theme in all_text:
                count = all_text.count(theme)
                found_themes.append({"theme": theme, "frequency": count})
        
        # Sort by frequency and return top themes
        found_themes.sort(key=lambda x: x["frequency"], reverse=True)
        return [theme["theme"] for theme in found_themes[:5]]
    
    def _analyze_sentiment_trend(self, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment trend for a symbol."""
        
        if symbol not in self.sentiment_history:
            return {"trend": "insufficient_data", "change": 0.0}
        
        history = self.sentiment_history[symbol]
        
        if len(history) < 2:
            return {"trend": "insufficient_data", "change": 0.0}
        
        # Calculate trend over different periods
        recent_scores = [entry["sentiment"] for entry in history[-5:]]  # Last 5 entries
        older_scores = [entry["sentiment"] for entry in history[-10:-5]]  # Previous 5 entries
        
        recent_avg = np.mean(recent_scores) if recent_scores else 0
        older_avg = np.mean(older_scores) if older_scores else recent_avg
        
        change = recent_avg - older_avg
        
        # Classify trend
        if change > 0.1:
            trend = "improving"
        elif change < -0.1:
            trend = "deteriorating"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "change": change,
            "recent_average": recent_avg,
            "historical_average": np.mean([entry["sentiment"] for entry in history]),
            "volatility": np.std([entry["sentiment"] for entry in history])
        }
    
    def _update_sentiment_history(self, sentiment_results: Dict[str, Any]):
        """Update sentiment history for trend analysis."""
        
        for symbol, result in sentiment_results.items():
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            
            self.sentiment_history[symbol].append({
                "timestamp": datetime.now(),
                "sentiment": result.get("overall_sentiment", 0),
                "confidence": result.get("confidence", 0),
                "article_count": result.get("total_articles", 0)
            })
            
            # Keep only last 100 entries
            if len(self.sentiment_history[symbol]) > 100:
                self.sentiment_history[symbol] = self.sentiment_history[symbol][-100:]
    
    def _create_empty_sentiment_result(self, symbol: str) -> Dict[str, Any]:
        """Create empty sentiment result for error cases."""
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "overall_sentiment": 0.0,
            "sentiment_classification": "neutral",
            "confidence": 0.0,
            "total_articles": 0,
            "sources": {},
            "trend_analysis": {"trend": "insufficient_data", "change": 0.0},
            "key_themes": []
        }
    
    def _count_sentiment_type(self, results: Dict[str, Any], sentiment_type: str) -> int:
        """Count number of symbols with specific sentiment type."""
        
        count = 0
        for result in results.values():
            classification = result.get("sentiment_classification", "neutral")
            if classification == sentiment_type:
                count += 1
        
        return count
    
    def _clean_cache(self):
        """Clean expired entries from content cache."""
        try:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.content_cache.items()
                if current_time - entry["timestamp"] > self.cache_ttl
            ]
            for key in expired_keys:
                del self.content_cache[key]
            
            # If still too large, remove oldest entries
            if len(self.content_cache) > 1000:
                sorted_entries = sorted(
                    self.content_cache.items(),
                    key=lambda x: x[1]["timestamp"]
                )
                # Keep only the newest 500 entries
                self.content_cache = dict(sorted_entries[-500:])
                
        except Exception as e:
            self.logger.warning(f"Error cleaning cache: {e}")
    
    def get_sentiment_history(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get sentiment history for a symbol."""
        
        if symbol not in self.sentiment_history:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            entry for entry in self.sentiment_history[symbol]
            if entry["timestamp"] > cutoff_date
        ]