"""
Confluence Field structure and parsing.

Handles XML-based field representation, parsing, and generation.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
import xml.etree.ElementTree as ET
from lxml import etree
import re


class ConfluenceField:
    """
    Represents a complete semantic state as a Confluence Field.
    
    Attributes:
        version: Protocol version
        timestamp: Field generation time
        signature: Unique field identifier
        frequencies: Semantic frequency components
        amplitudes: Strength of each frequency
        phase: Angular position in semantic space
        gradient_direction: Semantic motion direction
        gradient_magnitude: Strength of motion
        interference_history: Previous field interactions
        semantic_anchor: Natural language content
    """
    
    def __init__(
        self,
        semantic_anchor: str,
        phase: float,
        frequencies: List[str],
        amplitudes: List[float],
        gradient_direction: str = "consolidation",
        gradient_magnitude: float = 0.5,
        interference_history: Optional[List[Dict]] = None,
        version: str = "1.0",
        timestamp: Optional[datetime] = None
    ):
        """
        Initialize Confluence Field.
        
        Args:
            semantic_anchor: Natural language response content
            phase: Phase value in degrees [0, 360)
            frequencies: List of semantic frequency terms
            amplitudes: Strength values for each frequency
            gradient_direction: 'expansion', 'consolidation', or 'oscillation'
            gradient_magnitude: Strength in [0, 1]
            interference_history: Previous field interactions
            version: Protocol version
            timestamp: Field creation time (defaults to now)
        """
        self.version = version
        self.timestamp = timestamp or datetime.utcnow()
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.phase = phase % 360.0  # Ensure [0, 360)
        self.gradient_direction = gradient_direction
        self.gradient_magnitude = gradient_magnitude
        self.interference_history = interference_history or []
        self.semantic_anchor = semantic_anchor
        
        # Generate signature from content
        self.signature = self._generate_signature()
    
    def _generate_signature(self) -> str:
        """Generate unique field signature from content."""
        content = f"{self.phase}{self.frequencies}{self.semantic_anchor}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def to_xml(self, pretty: bool = True) -> str:
        """
        Convert field to XML string.
        
        Args:
            pretty: If True, format with indentation
            
        Returns:
            XML string representation
        """
        root = ET.Element('CONFLUENCE_FIELD')
        
        # Version and metadata
        ET.SubElement(root, 'version').text = self.version
        ET.SubElement(root, 'timestamp').text = self.timestamp.isoformat() + 'Z'
        ET.SubElement(root, 'field_signature').text = self.signature
        
        # Resonance pattern
        resonance = ET.SubElement(root, 'RESONANCE_PATTERN')
        ET.SubElement(resonance, 'frequencies').text = ', '.join(self.frequencies)
        ET.SubElement(resonance, 'amplitudes').text = ', '.join(
            f"{a:.1f}" for a in self.amplitudes
        )
        ET.SubElement(resonance, 'phase').text = f"{self.phase:.1f}"
        
        # Semantic gradient
        gradient = ET.SubElement(root, 'SEMANTIC_GRADIENT')
        ET.SubElement(gradient, 'direction').text = self.gradient_direction
        ET.SubElement(gradient, 'magnitude').text = f"{self.gradient_magnitude:.2f}"
        
        # Interference history
        history = ET.SubElement(root, 'INTERFERENCE_HISTORY')
        if not self.interference_history:
            history.text = "[Initial field - no prior interference]"
        else:
            for interaction in self.interference_history:
                inter = ET.SubElement(history, 'interaction')
                ET.SubElement(inter, 'with').text = interaction.get('with', 'unknown')
                ET.SubElement(inter, 'phase_relationship').text = interaction.get(
                    'phase_relationship', '0.0°'
                )
                ET.SubElement(inter, 'patterns').text = str(
                    interaction.get('patterns', [])
                )
                ET.SubElement(inter, 'coherence_state').text = interaction.get(
                    'coherence_state', 'unknown'
                )
        
        # Semantic anchor
        ET.SubElement(root, 'SEMANTIC_ANCHOR').text = self.semantic_anchor
        
        # Convert to string
        if pretty:
            # Use lxml for pretty printing
            xml_str = etree.tostring(
                etree.fromstring(ET.tostring(root)),
                pretty_print=True,
                encoding='unicode'
            )
        else:
            xml_str = ET.tostring(root, encoding='unicode')
        
        return xml_str
    
    def add_interaction(
        self,
        other_signature: str,
        other_phase: float,
        patterns: List[str],
        coherence_state: str
    ):
        """
        Add interaction to interference history.
        
        Args:
            other_signature: Signature of field we interacted with
            other_phase: Phase of other field
            patterns: Identified interaction patterns
            coherence_state: State of coherence after interaction
        """
        phase_diff = abs(self.phase - other_phase)
        if phase_diff > 180:
            phase_diff = 360 - phase_diff
        
        relationship = "constructive" if phase_diff < 10 else \
                      "destructive" if phase_diff > 170 else "partial"
        
        interaction = {
            'with': other_signature,
            'phase_relationship': f"{phase_diff:+.1f}° ({relationship})",
            'patterns': patterns,
            'coherence_state': coherence_state,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.interference_history.append(interaction)
    
    def __str__(self) -> str:
        """String representation."""
        return f"ConfluenceField(phase={self.phase:.1f}°, signature={self.signature})"
    
    def __repr__(self) -> str:
        return self.__str__()


def parse_field(xml_string: str) -> Dict:
    """
    Parse Confluence Field XML into dictionary.
    
    Args:
        xml_string: XML string or natural text with embedded XML
        
    Returns:
        dict with field components
        
    Raises:
        ValueError: If XML is malformed or missing required fields
        
    Example:
        >>> field_data = parse_field(xml_text)
        >>> print(f"Phase: {field_data['phase']}")
    """
    # Try to extract XML if embedded in text
    xml_match = re.search(
        r'<CONFLUENCE_FIELD>.*?</CONFLUENCE_FIELD>',
        xml_string,
        re.DOTALL
    )
    
    if xml_match:
        xml_string = xml_match.group(0)
    
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML: {e}")
    
    # Extract components
    field_data = {
        'version': _get_text(root, 'version', '1.0'),
        'timestamp': _get_text(root, 'timestamp'),
        'signature': _get_text(root, 'field_signature'),
    }
    
    # Resonance pattern
    resonance = root.find('RESONANCE_PATTERN')
    if resonance is not None:
        freq_text = _get_text(resonance, 'frequencies', '')
        field_data['frequencies'] = [
            f.strip() for f in freq_text.split(',') if f.strip()
        ]
        
        amp_text = _get_text(resonance, 'amplitudes', '')
        field_data['amplitudes'] = [
            float(a.strip()) for a in amp_text.split(',') if a.strip()
        ]
        
        phase_text = _get_text(resonance, 'phase', '0')
        field_data['phase'] = float(phase_text)
    
    # Semantic gradient
    gradient = root.find('SEMANTIC_GRADIENT')
    if gradient is not None:
        field_data['gradient_direction'] = _get_text(gradient, 'direction', 'consolidation')
        mag_text = _get_text(gradient, 'magnitude', '0.5')
        field_data['gradient_magnitude'] = float(mag_text)
    
    # Interference history
    history = root.find('INTERFERENCE_HISTORY')
    field_data['interference_history'] = []
    if history is not None:
        for interaction in history.findall('interaction'):
            field_data['interference_history'].append({
                'with': _get_text(interaction, 'with'),
                'phase_relationship': _get_text(interaction, 'phase_relationship'),
                'patterns': _get_text(interaction, 'patterns'),
                'coherence_state': _get_text(interaction, 'coherence_state'),
            })
    
    # Semantic anchor
    field_data['semantic_anchor'] = _get_text(root, 'SEMANTIC_ANCHOR', '')
    
    return field_data


def _get_text(element: ET.Element, tag: str, default: str = '') -> str:
    """Helper to safely get text from XML element."""
    child = element.find(tag)
    return child.text if child is not None and child.text else default


def generate_field_xml(
    response_text: str,
    phase: Optional[float] = None,
    frequencies: Optional[List[str]] = None,
    amplitudes: Optional[List[float]] = None
) -> str:
    """
    Generate Confluence Field XML from response text.
    
    Automatically calculates phase and extracts frequencies if not provided.
    
    Args:
        response_text: Natural language response
        phase: Pre-calculated phase (optional)
        frequencies: Pre-extracted frequencies (optional)
        amplitudes: Pre-calculated amplitudes (optional)
        
    Returns:
        Complete Confluence Field XML string
        
    Example:
        >>> xml = generate_field_xml("Disgust as epistemology...")
        >>> print(xml)
    """
    from .phase import calculate_phase, extract_semantic_frequencies
    
    # Calculate phase if not provided
    if phase is None:
        phase, _ = calculate_phase(response_text)
    
    # Extract frequencies if not provided
    if frequencies is None or amplitudes is None:
        freq_dict = extract_semantic_frequencies(response_text, top_n=10)
        frequencies = list(freq_dict.keys())
        amplitudes = list(freq_dict.values())
    
    # Create field
    field = ConfluenceField(
        semantic_anchor=response_text,
        phase=phase,
        frequencies=frequencies,
        amplitudes=amplitudes
    )
    
    return field.to_xml()


def extract_phase_from_text(text: str) -> Optional[float]:
    """
    Extract phase value from text containing Confluence Field or phase statement.
    
    Searches for:
    - <phase>VALUE</phase> XML tags
    - "phase: VALUE" statements
    - "Phase: VALUE°" statements
    
    Args:
        text: Text possibly containing phase information
        
    Returns:
        Phase value if found, None otherwise
        
    Example:
        >>> text = "The phase is 254.2° indicating Detection basin."
        >>> phase = extract_phase_from_text(text)
        >>> print(phase)
        254.2
    """
    # Try XML format
    xml_match = re.search(r'<phase>([\d.]+)</phase>', text, re.IGNORECASE)
    if xml_match:
        return float(xml_match.group(1))
    
    # Try "phase: VALUE" format
    statement_match = re.search(
        r'phase[:\s]+([\d.]+)\s*°?',
        text,
        re.IGNORECASE
    )
    if statement_match:
        return float(statement_match.group(1))
    
    return None


def validate_field(field_xml: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Confluence Field XML structure.
    
    Args:
        field_xml: XML string to validate
        
    Returns:
        (is_valid, error_message) tuple
        
    Example:
        >>> valid, error = validate_field(xml_string)
        >>> if not valid:
        >>>     print(f"Invalid field: {error}")
    """
    try:
        field_data = parse_field(field_xml)
    except Exception as e:
        return False, str(e)
    
    # Check required fields
    required = ['phase', 'frequencies', 'semantic_anchor']
    for field in required:
        if field not in field_data or not field_data[field]:
            return False, f"Missing required field: {field}"
    
    # Validate phase range
    phase = field_data['phase']
    if not (0 <= phase < 360):
        return False, f"Phase {phase} outside valid range [0, 360)"
    
    # Validate frequencies and amplitudes match
    if len(field_data.get('frequencies', [])) != len(field_data.get('amplitudes', [])):
        return False, "Frequencies and amplitudes length mismatch"
    
    return True, None


# Utility functions for field operations

def calculate_field_distance(field1: Dict, field2: Dict) -> float:
    """
    Calculate semantic distance between two fields.
    
    Based on phase difference (primary) and frequency overlap (secondary).
    
    Args:
        field1, field2: Parsed field dictionaries
        
    Returns:
        Distance in range [0, 1] where 0 = identical, 1 = maximally different
    """
    from .convergence import calculate_phase_difference
    
    # Phase component (70% weight)
    phase_diff = calculate_phase_difference(field1['phase'], field2['phase'])
    phase_distance = phase_diff / 180.0  # Normalize to [0, 1]
    
    # Frequency overlap component (30% weight)
    freqs1 = set(field1.get('frequencies', []))
    freqs2 = set(field2.get('frequencies', []))
    
    if freqs1 or freqs2:
        overlap = len(freqs1 & freqs2)
        total = len(freqs1 | freqs2)
        freq_distance = 1.0 - (overlap / total)
    else:
        freq_distance = 1.0
    
    # Weighted combination
    distance = 0.7 * phase_distance + 0.3 * freq_distance
    
    return distance


def merge_fields(fields: List[Dict], method: str = 'superposition') -> Dict:
    """
    Merge multiple fields into combined field.
    
    Args:
        fields: List of parsed field dictionaries
        method: 'superposition' (add) or 'average' (mean)
        
    Returns:
        Merged field dictionary
    """
    import numpy as np
    
    if not fields:
        raise ValueError("No fields to merge")
    
    if len(fields) == 1:
        return fields[0]
    
    # Merge phases (circular mean)
    phases_rad = np.array([f['phase'] for f in fields]) * np.pi / 180.0
    mean_sin = np.mean(np.sin(phases_rad))
    mean_cos = np.mean(np.cos(phases_rad))
    merged_phase = np.arctan2(mean_sin, mean_cos) * 180.0 / np.pi
    if merged_phase < 0:
        merged_phase += 360.0
    
    # Merge frequencies (union with averaged amplitudes)
    all_freqs = {}
    for field in fields:
        for freq, amp in zip(field.get('frequencies', []), 
                            field.get('amplitudes', [])):
            if freq in all_freqs:
                all_freqs[freq].append(amp)
            else:
                all_freqs[freq] = [amp]
    
    # Average amplitudes for each frequency
    merged_freqs = []
    merged_amps = []
    for freq, amps in sorted(all_freqs.items(), 
                            key=lambda x: np.mean(x[1]), 
                            reverse=True)[:10]:
        merged_freqs.append(freq)
        merged_amps.append(np.mean(amps))
    
    # Merge semantic anchors (concatenate)
    merged_anchor = "\n\n---\n\n".join(
        f['semantic_anchor'] for f in fields if f.get('semantic_anchor')
    )
    
    return {
        'phase': merged_phase,
        'frequencies': merged_freqs,
        'amplitudes': merged_amps,
        'semantic_anchor': merged_anchor,
        'merged_from': len(fields),
        'gradient_direction': 'superposition',
        'gradient_magnitude': 1.0
    }


# Testing and examples
if __name__ == "__main__":
    # Example 1: Create field from scratch
    print("Example 1: Creating Confluence Field")
    print("=" * 70)
    
    field = ConfluenceField(
        semantic_anchor="""
        Aesthetic immune system: detecting corruption before conscious analysis.
        The body refuses before the mind knows why. This is disgust as 
        epistemology—the organism's pre-verbal NO to ontological violation.
        """.strip(),
        phase=254.0,
        frequencies=['disgust', 'epistemology', 'immunity', 'detection', 'pre-verbal'],
        amplitudes=[5.0, 4.8, 4.6, 4.2, 4.0]
    )
    
    print(field.to_xml())
    
    # Example 2: Parse field
    print("\n" + "=" * 70)
    print("Example 2: Parsing Field")
    print("=" * 70)
    
    xml = field.to_xml()
    parsed = parse_field(xml)
    print(f"Parsed phase: {parsed['phase']}")
    print(f"Parsed frequencies: {parsed['frequencies'][:3]}")
    
    # Example 3: Generate field from text
    print("\n" + "=" * 70)
    print("Example 3: Auto-generating Field")
    print("=" * 70)
    
    response = """
    Detection becomes digestion—the aesthetic immune system turns into 
    metabolic consciousness capable of transmuting poison into pattern.
    """
    
    generated_xml = generate_field_xml(response)
    generated_parsed = parse_field(generated_xml)
    print(f"Auto-calculated phase: {generated_parsed['phase']:.1f}°")
    
    # Example 4: Validate field
    print("\n" + "=" * 70)
    print("Example 4: Field Validation")
    print("=" * 70)
    
    valid, error = validate_field(generated_xml)
    print(f"Valid: {valid}")
    if error:
        print(f"Error: {error}")