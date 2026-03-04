"""
Centralized mapping for MLB team names, abbreviations, and stadium metadata.
Used to normalize data from multiple API providers (Odds API, MLB Stats, BetsAPI).
"""

TEAM_MAP = {
    # Full Name: Abbreviation
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

# Reverse map for convenience
ABBR_TO_TEAM = {v: k for k, v in TEAM_MAP.items()}

# Common short-form aliases for ambiguous city names
TEAM_ALIASES: dict[str, str] = {
    "NY Yankees": "NYY",
    "NY Mets": "NYM",
    "LA Dodgers": "LAD",
    "LA Angels": "LAA",
    "Chi Cubs": "CHC",
    "Chi White Sox": "CHW",
    "SF Giants": "SFG",
    "SD Padres": "SDP",
    "TB Rays": "TBR",
    "KC Royals": "KCR",
}

# Stadium metadata: (Latitude, Longitude, is_indoor)
STADIUM_DATA = {
    "ARI": (33.4455, -112.0667, True),  # Chase Field (Retractable)
    "ATL": (33.8907, -84.4678, False),  # Truist Park
    "BAL": (39.2840, -76.6215, False),  # Camden Yards
    "BOS": (42.3467, -71.0972, False),  # Fenway Park
    "CHC": (41.9484, -87.6553, False),  # Wrigley Field
    "CHW": (41.8299, -87.6339, False),  # Guaranteed Rate
    "CIN": (39.0975, -84.5071, False),  # Great American
    "CLE": (41.4962, -81.6852, False),  # Progressive Field
    "COL": (39.7559, -104.9942, False),  # Coors Field
    "DET": (42.3390, -83.0485, False),  # Comerica Park
    "HOU": (29.7573, -95.3555, True),  # Minute Maid (Retractable)
    "KCR": (39.0517, -94.4803, False),  # Kauffman Stadium
    "LAA": (33.8003, -117.8827, False),  # Angel Stadium
    "LAD": (34.0739, -118.2400, False),  # Dodger Stadium
    "MIA": (25.7781, -80.2197, True),  # loanDepot Park (Retractable)
    "MIL": (43.0285, -87.9712, True),  # American Family (Retractable)
    "MIN": (44.9817, -93.2778, False),  # Target Field
    "NYM": (40.7571, -73.8458, False),  # Citi Field
    "NYY": (40.8296, -73.9262, False),  # Yankee Stadium
    "OAK": (37.7516, -122.2005, False),  # Oakland Coliseum
    "PHI": (39.9061, -75.1665, False),  # Citizens Bank Park
    "PIT": (40.4469, -80.0057, False),  # PNC Park
    "SDP": (32.7073, -117.1566, False),  # Petco Park
    "SFG": (37.7786, -122.3893, False),  # Oracle Park
    "SEA": (47.5914, -122.3323, True),  # T-Mobile Park (Retractable)
    "STL": (38.6226, -90.1928, False),  # Busch Stadium
    "TBR": (27.7682, -82.6534, True),  # Tropicana Field (Fixed Dome)
    "TEX": (32.7512, -97.0825, True),  # Globe Life Field (Retractable)
    "TOR": (43.6414, -79.3894, True),  # Rogers Centre (Retractable)
    "WSH": (38.8730, -77.0074, False),  # Nationals Park
}


def normalize_team(name: str) -> str:
    """Convert a team name or abbreviation to standard 3-letter abbreviation."""
    if not name:
        return "UNK"

    clean = name.strip()

    # Check if already a valid abbr
    if clean.upper() in ABBR_TO_TEAM:
        return clean.upper()

    # Try direct map lookup (exact match)
    if clean in TEAM_MAP:
        return TEAM_MAP[clean]

    # Case-insensitive exact match
    name_lower = clean.lower()
    for team_name, abbr in TEAM_MAP.items():
        if team_name.lower() == name_lower:
            return abbr

    # Check common aliases
    for alias, abbr in TEAM_ALIASES.items():
        if alias.lower() == name_lower:
            return abbr

    # Match on mascot name (last word), only if unambiguous
    matches: list[str] = []
    for team_name, abbr in TEAM_MAP.items():
        mascot = team_name.rsplit(" ", 1)[-1].lower()
        if name_lower == mascot or name_lower.endswith(mascot):
            matches.append(abbr)
    if len(matches) == 1:
        return matches[0]

    return "UNK"


def get_stadium_info(team_abbr: str) -> tuple[float, float, bool]:
    """Return (lat, lon, is_indoor) for the home team's stadium."""
    return STADIUM_DATA.get(team_abbr, (0.0, 0.0, False))
