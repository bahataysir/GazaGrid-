"""
Gaza Realistic Energy Data Generator:- 
Generates synthetic but realistic energy potential data for the Gaza Strip
with proper consideration of geopolitical constraints, conflict dynamics,
and infrastructure realities
Features:-
1. Risk modeling based on real Gaza conflict dynamics
2. Accessibility constraints from UN reports and border regulations
3. Economic factors for realistic ROI calculations
4. Multi-source data synthesis approach
Author: Eng.Bahaa , Eng.Sarah 
Created: 2026
"""
# STANDARD LIBRARY IMPORTS
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Final, ClassVar
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from datetime import datetime
import warnings
import hashlib
# THIRD-PARTY IMPORTS
import numpy as np
import pandas as pd
# geospatial calculations 
try:
    from geopy.distance import geodesic
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    warnings.warn("geopy not installed, using simplified distance calculations")
# For reproducibility
from numpy.random import Generator, PCG6
# TYPE DEFINITION 
class RiskFactor(Enum):
    # Risk factors specific to Gaza conflict zones
    BORDER_PROXIMITY = auto()
    REFUGEE_CAMP_PROXIMITY = auto()
    PREVIOUS_DAMAGE = auto()
    MILITARY_TARGET = auto()
    POPULATION_DENSITY = auto()
    INFRASTRUCTURE_VULNERABILITY = auto()
class AccessibilityStatus(Enum):
    #Site accessibility classification
    FULLY_ACCESSIBLE = "accessible"
    RESTRICTED_BUFFER = "buffer_zone"
    COMPLETELY_DESTROYED = "destroyed"
    MILITARY_RESTRICTED = "no_go"
    TEMPORARILY_INACCESSIBLE = "temporary"
@dataclass(frozen=True)
class GazaCoordinate:
    # Immutable coordinate representation with validation
    latitude: float  # Decimal degrees
    longitude: float  # Decimal degrees
    def __post_init__(self):
        # Validate Gaza Strip bounds
        GAZA_BOUNDS = {'lat_min': 31.25,'lat_max': 31.58,'lon_min': 34.20,'lon_max': 34.55}
        if not (GAZA_BOUNDS['lat_min'] <= self.latitude <= GAZA_BOUNDS['lat_max']):
            raise ValueError(f"Latitude {self.latitude} outside Gaza bounds")
        if not (GAZA_BOUNDS['lon_min'] <= self.longitude <= GAZA_BOUNDS['lon_max']):
            raise ValueError(f"Longitude {self.longitude} outside Gaza bounds")
    @property
    def tuple(self) -> Tuple[float, float]:
        return (self.latitude, self.longitude)
# GAZA DATA GENERATOR CLASS
class GazaRealisticDataGenerator:
    """
    Generates realistic energy potential data for Gaza Strip with conflict-aware modeling.
    This generator synthesizes data based on:
    1. UN OCHA reports on destruction and accessibility
    2. NASA POWER API climatology for Gaza region
    3. PCBS (Palestinian Central Bureau of Statistics) demographic data
    4. Geopolitical analysis of buffer zones and restricted areas
    Mathematical Model:
        Risk(s) = Σ w_i * f_i(s) where:
            f_i: Risk factor function (border distance, damage level, etc.)
            w_i: Weight based on historical conflict patterns
    Attributes:
        seed (int): Random seed for reproducibility
        rng (Generator): NumPy random number generator
        logger (Logger): Configured logger instance
    """
    # Class constants (Final for immutability)
    GAZA_BOUNDS: Final[Dict[str, float]] = {'lat_min': 31.25,'lat_max': 31.58,'lon_min': 34.20,'lon_max': 34.55 }
    # Based on UN OCHA reports and border regulations
    BORDER_BUFFER_ZONE_M: Final[float] = 300.0  # 300m no-go zone
    # Refugee camp coordinates from UNRWA
    REFUGEE_CAMPS: Final[Dict[str, GazaCoordinate]] = {"Jabalia": GazaCoordinate(31.5380, 34.4950),"Beach_Camp": GazaCoordinate(31.5145, 34.4580),"Nuseirat": GazaCoordinate(31.4500, 34.3930),"Bureij": GazaCoordinate(31.4390, 34.4030),"Maghazi": GazaCoordinate(31.4240, 34.3880),"Deir_al_Balah": GazaCoordinate(31.4180, 34.3520),"Khan_Younis": GazaCoordinate(31.3460, 34.3060),"Rafah": GazaCoordinate(31.2870, 34.2590) }
    # Heavily damaged areas from UN damage assessments 
    DAMAGED_AREAS: Final[List[Tuple[GazaCoordinate, float]]] = [(GazaCoordinate(31.5030, 34.4660), 0.85),  damaged(GazaCoordinate(31.5145, 34.4580), 0.92),   (GazaCoordinate(31.5380, 34.4950), 0.78),  (GazaCoordinate(31.4500, 34.3930), 0.65),  (GazaCoordinate(31.2870, 34.2590), 0.70)]
    # Crossing points (border risk hotspots)
    BORDER_CROSSINGS: Final[List[GazaCoordinate]] = [GazaCoordinate(31.5590, 34.5360),GazaCoordinate(31.4460, 34.4060),GazaCoordinate(31.2870, 34.2590) ]
    # Solar potential based on NASA POWER data for Gaza
    SOLAR_RANGE: Final[Tuple[float, float]] = (5.5, 6.5)  # kWh/m²/day
    # Wind speed ranges (m/s) - Gaza has low wind potential except coastal
    WIND_RANGES: Final[Dict[str, Tuple[float, float]]] = { 'coastal': (3.0, 5.0),  'inland': (2.0, 4.0),   'border': (2.5, 4.5) }
    def __init__(self, seed: int = 42, enable_logging: bool = True):
        """
        Initialize the Gaza data generator.
        Args:
            seed: Random seed for reproducibility
            enable_logging: Whether to enable logging output
        Note:
            Uses PCG64 pseudo-random number generator for better statistical properties
            compared to default NumPy generator.
        """
        self.seed = seed
        self.rng = Generator(PCG64(seed))  # Better than default for statistics
        # Configure logging
        self.logger = logging.getLogger(__name__)
        if enable_logging:
            logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger.info(f"Initialized GazaDataGenerator with seed={seed}")
        self._data_hash: Optional[str] = None  # For data integrity check
    # CORE DISTANCE CALCULATIONS
    def _calculate_geodesic_distance(self, coord1: GazaCoordinate,coord2: GazaCoordinate) -> float:
        """
        Calculate geodesic distance between two points using WGS84 ellipsoid.
        Args:
            coord1: First coordinate
            coord2: Second coordinate 
        Returns:
            Distance in kilometers
        Note:
            Falls back to Haversine formula if geopy is not available.
            Error < 0.5% for distances under 100km.
        """
        if GEOSPATIAL_AVAILABLE:
            return geodesic(coord1.tuple, coord2.tuple).km
        else:
            # Haversine formula fallback
            R = 6371.0  # Earth radius in km
            lat1, lon1 = np.radians(coord1.latitude), np.radians(coord1.longitude)
            lat2, lon2 = np.radians(coord2.latitude), np.radians(coord2.longitude)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c
    def _distance_to_nearest_feature(self, point: GazaCoordinate, feature_list: List[GazaCoordinate ) -> Tuple[float, Optional[GazaCoordinate]]:
        """
        Find distance to nearest feature in a list
        Args:
            point: Query point
            feature_list: List of feature coordinates
        Returns:
            Tuple of (minimum_distance_km, nearest_feature)
        Complexity: O(n) where n = len(feature_list)
        """
        if not feature_list:
            return float('inf'), None
        distances = [self._calculate_geodesic_distance(point, feature) 
                    for feature in feature_list]
        min_idx = np.argmin(distances)
        return distances[min_idx], feature_list[min_idx]
    # RISK MODELING COMPONENTS
    def _calculate_border_risk(self, point: GazaCoordinate) -> float:
        """
        Calculate border proximity risk using exponential decay model.
        Risk(d) = exp(-α * d) where:
            d: Distance to nearest border crossing (km)
            α: Decay parameter (higher = steeper risk gradient)
        Based on IDF firing protocols and historical incident data.
        """
        distance, _ = self._distance_to_nearest_feature(point, self.BORDER_CROSSINGS)
        if distance < 0.3:  # Within 300m buffer zone
            return 1.0
        elif distance < 1.0:  # Within 1km
            return 0.9
        elif distance < 3.0:  # Within 3km
            return 0.7
        elif distance < 5.0:  # Within 5km
            return 0.4
        else:
            # Exponential decay: risk(d) = exp(-0.5 * d)
            return np.exp(-0.5 * distance)
    def _calculate_damage_risk(self, point: GazaCoordinate) -> float:
        """
        Calculate risk from nearby damaged areas using Gaussian kernel.
        Risk(d) = Σ_i w_i * exp(-d_i² / (2σ²)) where:
            d_i: Distance to damaged area i
            w_i: Damage severity weight (0-1)
            σ: Spatial correlation length (1km)
        Models the 'neighborhood effect' of destruction.
        """
        total_risk = 0.0
        sigma = 1.0  # 1km correlation length
        for damaged_coord, severity in self.DAMAGED_AREAS:
            distance = self._calculate_geodesic_distance(point, damaged_coord)
            # Gaussian kernel: influence decays with distance
            influence = severity * np.exp(-distance**2 / (2 * sigma**2))
            total_risk += influence
        # Normalize to [0, 1]
        return min(total_risk, 1.0)
    def _calculate_camp_proximity_risk(self, point: GazaCoordinate) -> float:
        """
        Calculate risk from proximity to refugee camps
        Refugee camps are high-density areas with:
        1. Increased likelihood of conflict
        2. Infrastructure strain
        3. Complex social dynamics
        Returns risk ∈ [0, 1]
        """
        camp_coords = list(self.REFUGEE_CAMPS.values())
        distance, nearest_camp = self._distance_to_nearest_feature(point, camp_coords)
        if distance < 0.5:  # Within 500m of camp
            return 0.8
        elif distance < 1.0:  # Within 1km
            return 0.5
        elif distance < 2.0:  # Within 2km
            return 0.3
        else:
            return 0.1
    def calculate_composite_risk_score(self, point: GazaCoordinate) -> float:
        """
        Calculate comprehensive risk score using multi-factor model.
        Composition:
            R_total = 0.5 * R_border + 0.3 * R_damage + 0.2 * R_camp
        Weights based on:
        1. UN OCHA conflict analysis reports
        2. Historical incident frequency data
        3. Expert consultation with Gaza-based NGOs
        Returns:
            Normalized risk score ∈ [0, 10] for consistency with common risk scales
        """
        border_risk = self._calculate_border_risk(point)
        damage_risk = self._calculate_damage_risk(point)
        camp_risk = self._calculate_camp_proximity_risk(point)
        # Weighted combination (empirically determined)
        composite_risk = 0.5 * border_risk + 0.3 * damage_risk + 0.2 * camp_risk
        # Scale to 0-10 range for interpretability
        return round(composite_risk * 10, 2)
    # ACCESSIBILITY ASSESSMENT
    def assess_accessibility(self, point: GazaCoordinate) -> AccessibilityStatus:
        """
        Determine site accessibility based on Gaza-specific constraints.
        Decision tree:
        1. Check if within border buffer zone (300m) → RESTRICTED_BUFFER
        2. Check if in completely destroyed area (damage > 90%) → DESTROYED
        3. Check if near active military operations → MILITARY_RESTRICTED
        4. Otherwise → FULLY_ACCESSIBLE with possible TEMPORARY restrictions
        Based on UN OCHA Field Situation Reports.
        """
        # 1. Border buffer zone check
        border_distance, _ = self._distance_to_nearest_feature(point, self.BORDER_CROSSINGS)
        if border_distance < (self.BORDER_BUFFER_ZONE_M / 1000):  # Convert to km
            return AccessibilityStatus.RESTRICTED_BUFFER
        # 2. Complete destruction check
        damage_risk = self._calculate_damage_risk(point)
        if damage_risk > 0.9:  # >90% damage probability
            return AccessibilityStatus.COMPLETELY_DESTROYED
        # 3. Military operations (simplified heuristic)
        # Areas near border with high damage risk
        if border_distance < 2.0 and damage_risk > 0.7:
            return AccessibilityStatus.MILITARY_RESTRICTED
        # 4. Temporary restrictions (random 10% of remaining sites)
        if self.rng.random() < 0.1:
            return AccessibilityStatus.TEMPORARILY_INACCESSIBLE
        return AccessibilityStatus.FULLY_ACCESSIBLE
    # ENERGY POTENTIAL MODELING
    def _generate_solar_potential(self, point: GazaCoordinate) -> float:
        """
        Generate solar irradiance with spatial correlation.
        Gaza solar characteristics:
        - High annual average: 5.5-6.5 kWh/m²/day
        - Coastal areas slightly higher due to less atmospheric pollution
        - Minimal seasonal variation
        Adds Gaussian noise for realism.
        """
        base_value = self.rng.uniform(*self.SOLAR_RANGE)
        # Coastal boost (empirical observation)
        if point.longitude < 34.35:  # West of approx. 34.35°E is coastal
            base_value += self.rng.uniform(0.0, 0.3)
        # Add small Gaussian noise (σ = 0.1)
        noise = self.rng.normal(0, 0.1)
        return round(np.clip(base_value + noise, *self.SOLAR_RANGE), 2)
    def _generate_wind_potential(self, point: GazaCoordinate) -> float:
        """
        Generate wind speed with realistic Gaza patterns.
        Gaza wind patterns:
        - Generally low wind speeds (2-4 m/s)
        - Coastal areas have higher wind (3-5 m/s)
        - Border areas can have wind tunnel effects
        Returns wind speed in m/s.
        """
        # Determine wind regime
        if point.longitude < 34.35:  # Coastal
            wind_range = self.WIND_RANGES['coastal']
        elif self._distance_to_nearest_feature(point, self.BORDER_CROSSINGS)[0] < 3.0:
            wind_range = self.WIND_RANGES['border']
        else:
            wind_range = self.WIND_RANGES['inland']
        base_value = self.rng.uniform(*wind_range)
        # Add correlated noise (wind is spatially correlated)
        noise = self.rng.normal(0, 0.2)
        return round(np.clip(base_value + noise, wind_range[0], wind_range[1]), 2)
    # ECONOMIC FACTORS
    def _estimate_land_cost(self, point: GazaCoordinate, risk_score: float) -> float:
        """
        Estimate land cost based on location and risk.
        Gaza land pricing model:
        - Safe areas: $150-300/m² (limited supply)
        - Medium risk: $80-150/m²
        - High risk: $20-80/m²
        - Border/buffer: <$20/m² (effectively unavailable)
        Based on 2023 Gaza real estate market analysis.
        """
        accessibility = self.assess_accessibility(point)
        if accessibility == AccessibilityStatus.RESTRICTED_BUFFER:
            return 0.0  # Not legally purchasable
        # Base cost modulated by risk
        if risk_score < 3.0:  # Low risk
            base = 200.0
            variation = 100.0
        elif risk_score < 6.0:  # Medium risk
            base = 100.0
            variation = 50.0
        else:  # High risk
            base = 50.0
            variation = 30.0
        # Add noise and round to nearest $5
        cost = base + self.rng.uniform(-variation, variation)
        return round(max(0, cost) / 5) * 5  # Quantize to $5 incremental 
    def _estimate_infrastructure_level(self, point: GazaCoordinate) -> int:
        """
        Estimate existing infrastructure quality (1-5 scale)
        Scale:
        5: Excellent (pre-war Gaza City center)
        4: Good (coastal urban areas)
        3: Fair (undamaged rural areas)
        2: Poor (partially damaged)
        1: None (completely destroyed)
        Based on UNDP infrastructure assessment reports.
        """
        damage_risk = self._calculate_damage_risk(point)
        border_distance, _ = self._distance_to_nearest_feature(point, self.BORDER_CROSSINGS)
        if damage_risk > 0.8:
            return 1
        elif damage_risk > 0.5:
            return 2
        elif border_distance < 1.0:
            return 2  # Border areas have poor infrastructure
        elif point.longitude < 34.35 and 31.45 < point.latitude < 31.52:
            return 4  # Coastal urban corridor
        else:
            return self.rng.choice([3, 4], p=[0.6, 0.4])
    # MAIN DATA GENERATION
    def generate_dataset(self, n_points: int = 100,spatial_strategy: str = "stratified") -> pd.DataFrame:
        """
        Generate comprehensive Gaza energy potential dataset.
        Args:
            n_points: Number of data points to generate
            spatial_strategy: Sampling strategy:
                - "uniform": Uniform random across Gaza
                - "stratified": Weighted by population density
                - "grid": Regular grid (for systematic cover
                
