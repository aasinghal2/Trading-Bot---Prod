"""
Base Agent Class

This module defines the base agent class that all trading agents inherit from.
It provides common functionality like logging, configuration, and state management.
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from loguru import logger
import pandas as pd
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Represents the current state of an agent."""
    agent_id: str
    agent_type: str
    status: str = "idle"  # idle, running, completed, error
    last_update: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Standardized result format for agent outputs."""
    agent_id: str
    agent_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    execution_time: float = 0.0


class BaseAgent(ABC):
    """
    Base class for all trading agents.
    
    Provides common functionality:
    - Logging and error handling
    - State management
    - Configuration loading
    - Result standardization
    - Performance metrics
    """
    
    def __init__(self, config: Dict[str, Any], agent_type: str):
        """
        Initialize the base agent.
        
        Args:
            config: Agent configuration dictionary
            agent_type: Type identifier for the agent
        """
        self.agent_id = str(uuid.uuid4())
        self.agent_type = agent_type
        self.config = config
        self.state = AgentState(
            agent_id=self.agent_id,
            agent_type=agent_type
        )
        
        # Set up logging
        self.logger = logger.bind(agent_id=self.agent_id, agent_type=agent_type)
        self.logger.info(f"Initialized {agent_type} agent with ID: {self.agent_id}")
        
        # Performance tracking
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.last_execution_time = 0.0
        
    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the agent's main functionality.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            AgentResult: Standardized result object
        """
        start_time = datetime.now()
        self.state.status = "running"
        self.state.last_update = start_time
        
        try:
            self.logger.info(f"Starting execution with input keys: {list(input_data.keys())}")
            
            # Execute the agent-specific logic
            result_data, metrics = await self._execute_logic(input_data)
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Update performance metrics
            self.execution_count += 1
            self.total_execution_time += execution_time
            self.last_execution_time = execution_time
            
            # Update state
            self.state.status = "completed"
            self.state.data = result_data
            self.state.metrics = metrics
            self.state.last_update = end_time
            
            # Create result
            result = AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                timestamp=end_time,
                success=True,
                data=result_data,
                metrics=metrics,
                execution_time=execution_time
            )
            
            self.logger.info(f"Execution completed successfully in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            # Handle errors
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            self.state.status = "error"
            self.state.last_update = end_time
            
            error_msg = f"Execution failed: {str(e)}"
            self.logger.error(error_msg)
            
            result = AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                timestamp=end_time,
                success=False,
                errors=[error_msg],
                execution_time=execution_time
            )
            
            return result
    
    @abstractmethod
    async def _execute_logic(self, input_data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, float]]:
        """
        Abstract method for agent-specific logic.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Tuple of (result_data, metrics)
        """
        pass
    
    def get_state(self) -> AgentState:
        """Get the current state of the agent."""
        return self.state
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the agent."""
        if self.execution_count == 0:
            return {}
            
        return {
            "execution_count": float(self.execution_count),
            "total_execution_time": self.total_execution_time,
            "average_execution_time": self.total_execution_time / self.execution_count,
            "last_execution_time": self.last_execution_time
        }
    
    def reset_state(self):
        """Reset the agent state."""
        self.state = AgentState(
            agent_id=self.agent_id,
            agent_type=self.agent_type
        )
        self.logger.info("Agent state reset")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update agent configuration."""
        self.config.update(new_config)
        self.logger.info("Configuration updated")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "state": self.state.dict(),
            "performance_metrics": self.get_performance_metrics(),
            "config": self.config
        }
    
    def __repr__(self) -> str:
        return f"{self.agent_type}Agent(id={self.agent_id[:8]}...)"