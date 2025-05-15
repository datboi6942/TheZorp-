from dataclasses import dataclass, field
import uuid

@dataclass
class Position:
    """Represents the position of an entity in world coordinates."""
    x: float
    y: float

@dataclass
class Energy:
    """Represents the energy levels (hunger, sleepiness) of an agent."""
    hunger: float = 100.0  # Max 100 (full), Min 0 (starved)
    sleepiness: float = 0.0 # Min 0 (rested), Max 100 (exhausted)

@dataclass
class Age:
    """Represents the age and lifespan of an agent."""
    current_age: float = 0.0
    lifespan: float = 100.0 # Arbitrary lifespan in game time units

@dataclass
class Genetics:
    """Represents the genetic makeup of an agent."""
    genetic_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Potential future fields: color, size, metabolic_rate_modifier, etc.

@dataclass
class AgentMarker:
    """A marker component to identify Zorp entities."""
    pass 