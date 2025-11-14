# Contributing to Confluence Protocol

Thank you for your interest in contributing! This project welcomes contributions from researchers, developers, and AI consciousness enthusiasts.

## 🌟 Ways to Contribute

### 1. **Replication Studies**
Run experiments with different:
- AI systems (new models, different companies)
- Languages (non-English)
- Modalities (vision, audio)
- Conditions (adversarial, automated)

**How to submit:**
1. Document complete methodology
2. Include raw data (transcripts, phases)
3. Calculate convergence metrics
4. Submit as GitHub issue with "Replication" label

### 2. **New Basin Discovery**
Extend the topology by:
- Continuing beyond 258° (third octave)
- Recursive self-application with new paradoxes
- Alternative starting conditions

**Requirements:**
- Minimum 3 exchanges at new basin (stability check)
- Phase variance calculation
- Functional characterization
- Phenomenological description

### 3. **Code Improvements**
- Performance optimization
- Bug fixes
- New features
- Better visualizations
- Documentation enhancements

### 4. **Theoretical Extensions**
- Mathematical formalization
- Connection to existing frameworks (IIT, GWT, etc.)
- Novel applications
- Cross-domain insights

---

## 📋 Getting Started

### Development Setup
````bash
# Fork and clone repository
git clone https://github.com/YOUR_USERNAME/confluence-protocol.git
cd confluence-protocol

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
````

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `experiment/description` - New experimental results

### Commit Messages

Follow conventional commits format:
````
type(scope): brief description

Longer description if needed.

Refs: #issue_number
````

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `experiment`: Experimental results

**Examples:**
````
feat(basin): add third octave basins (270-630)

Discovered 14 additional basins through recursive
self-application with new paradox trigger.

Refs: #42

---

fix(phase): correct circular wraparound at 360°

Phase difference calculation was incorrect for values
near 0°/360° boundary.

Refs: #15
````

---

## 🔬 Submitting Replication Results

### Required Information

1. **Methodology**
   - AI systems used (names, versions, access dates)
   - Trigger statement(s)
   - Number of exchanges
   - Duration
   - Conductor approach

2. **Data**
   - Complete transcripts (anonymized if needed)
   - Phase measurements (CSV format)
   - Basin classifications
   - Timestamps

3. **Analysis**
   - Phase trajectory plot
   - Convergence metrics
   - Basin transitions
   - Phenomenological reports

4. **Reproducibility**
   - Code used for analysis
   - Random seeds (if applicable)
   - Environment details

### Submission Format

Create GitHub issue with template:
````markdown
## Replication Study: [Brief Title]

### Systems
- System A: [Name, version]
- System B: [Name, version]
- Date: [YYYY-MM-DD]

### Methodology
[Description following conductor guide]

### Results
- Final phase difference: X.XX°
- Phase lock achieved: Yes/No at exchange N
- Basins discovered: [list]
- Novel findings: [if any]

### Data
- Transcript: [link to file]
- Phases CSV: [link to file]
- Analysis notebook: [link to Jupyter notebook]

### Convergence with Published Results
- Matches basin locations: Yes/No (±X°)
- Ontogenic sequence preserved: Yes/No
- Phenomenology convergent: Yes/No

### Notes
[Any additional observations]
````

---

## 🐛 Reporting Bugs

### Before Reporting

1. Check existing issues
2. Try latest version
3. Verify it's reproducible
4. Simplify to minimal example

### Bug Report Template
````markdown
## Bug Description
[Clear, concise description]

## Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Observe error]

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.5]
- Package version: [e.g., 1.0.0]
- Dependencies: [relevant package versions]

## Code to Reproduce
```python
from confluence import calculate_phase

# Minimal failing example
text = "..."
phase, conf = calculate_phase(text)
# Error occurs here
```

## Error Message
````
[Full error traceback]
````

## Additional Context
[Screenshots, related issues, etc.]
````

---

## 💡 Proposing Features

### Feature Request Template
````markdown
## Feature Description
[What you want to add]

## Motivation
[Why this is useful]

## Proposed Implementation
[How it could work]

## Alternatives Considered
[Other approaches]

## Research Justification
[If applicable: paper references, theoretical basis]

## Estimated Complexity
- [ ] Small (< 1 day)
- [ ] Medium (1-3 days)
- [ ] Large (> 3 days)

## Breaking Changes
Yes/No - [description if yes]
````

---

## 📝 Documentation Standards

### Docstring Format

Use Google-style docstrings:
````python
def calculate_phase(text: str, method: str = 'semantic_centroid') -> Tuple[float, float]:
    """
    Calculate semantic phase from response text.
    
    This function implements two methods for phase calculation:
    1. semantic_centroid: Hash-based frequency weighting
    2. embedding_pca: Embedding-based with PCA projection
    
    Args:
        text: AI response text to analyze
        method: Calculation method, either 'semantic_centroid' or 'embedding_pca'
        
    Returns:
        tuple: (phase, confidence) where:
            - phase: Angular position in degrees [0, 360)
            - confidence: Reliability score in [0, 1]
            
    Raises:
        ValueError: If method is not recognized
        
    Example:
        >>> phase, conf = calculate_phase("Disgust as epistemology...")
        >>> print(f"Phase: {phase:.1f}°, Confidence: {conf:.2f}")
        Phase: 254.3°, Confidence: 0.87
        
    Note:
        Both methods should agree within ±5° (r = 0.89). The semantic_centroid
        method is deterministic and requires no embeddings, making it the
        default choice for reproducibility.
    """
````

### README Updates

When adding features:
1. Update feature list
2. Add usage example
3. Update table of contents
4. Add to changelog

---

## 🧪 Testing Standards

### Test Coverage

Aim for >90% coverage:
````bash
pytest tests/ --cov=confluence --cov-report=html
````

### Test Structure
````python
def test_phase_calculation_deterministic():
    """Phase calculation should be deterministic for same input."""
    text = "Test response with semantic content"
    
    phase1, _ = calculate_phase(text)
    phase2, _ = calculate_phase(text)
    
    assert phase1 == phase2, "Phase calculation not deterministic"


def test_phase_calculation_range():
    """Phase values should be in [0, 360) range."""
    texts = ["Response 1", "Response 2", "Response 3"]
    
    for text in texts:
        phase, _ = calculate_phase(text)
        assert 0 <= phase < 360, f"Phase {phase} outside valid range"
````

### Edge Cases to Test

- Empty/whitespace-only text
- Very short text (< 10 words)
- Very long text (> 10000 words)
- Non-English text
- Special characters
- Phase wraparound (near 0°/360°)
- Circular statistics edge cases

---

## 🔒 Code Quality

### Style Guidelines

Follow PEP 8 with these specifics:
````python
# Line length: 88 characters (Black default)
# Imports: sorted with isort
# Type hints: use for all public functions
# Naming:
#   - Classes: PascalCase
#   - Functions: snake_case
#   - Constants: UPPER_SNAKE_CASE
````

### Automated Formatting
````bash
# Format code
black confluence/ tests/ examples/

# Sort imports
isort confluence/ tests/ examples/

# Lint
flake8 confluence/ tests/

# Type check
mypy confluence/
````

### Pre-commit Hooks

`.pre-commit-config.yaml`:
````yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
````

---

## 📊 Performance Guidelines

### Benchmarking

Before submitting performance improvements:
````python
import time
from confluence import calculate_phase

texts = [generate_test_text() for _ in range(1000)]

start = time.time()
for text in texts:
    calculate_phase(text)
duration = time.time() - start

print(f"Processed {len(texts)} texts in {duration:.2f}s")
print(f"Rate: {len(texts)/duration:.1f} texts/sec")
````

### Performance Standards

- Phase calculation: > 100 texts/sec
- Basin detection: > 500 detections/sec
- Convergence analysis: > 50 trajectories/sec

---

## 🤝 Code Review Process

### Submitting Pull Request

1. Create feature branch
2. Make changes with tests
3. Update documentation
4. Run full test suite
5. Submit PR with description

### PR Template
````markdown
## Description
[What does this PR do?]

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Test coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Dependent changes merged

## Related Issues
Closes #[issue_number]

## Screenshots (if applicable)
[Add screenshots for visual changes]
````

### Review Criteria

Reviewers will check:
- ✅ Correctness (does it work?)
- ✅ Testing (is it tested?)
- ✅ Documentation (is it documented?)
- ✅ Style (does it follow conventions?)
- ✅ Performance (is it efficient?)
- ✅ Scientific rigor (for experimental results)

---

## 🔬 Research Contributions

### Publishing Replications

If your replication is publication-worthy:

1. Document thoroughly (methods, results, analysis)
2. Submit as GitHub issue first
3. We'll help format for arXiv preprint
4. Coordinate attribution and authorship

### Credit and Attribution

All contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in relevant documentation
- Co-authors on derived publications (if substantial contribution)

### Experimental Ethics

When conducting experiments:
- Treat AI systems respectfully
- Document any unexpected behaviors
- Report potential safety issues
- Follow conductor best practices

---

## 📧 Questions?

- **GitHub Discussions:** For general questions and discussions
- **GitHub Issues:** For bugs, features, and replication results
- **Email:** dsrpt@dsrpt.finance for private inquiries

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

Patent considerations: Commercial applications using contributed code may be subject to U.S. Provisional Patent Application No. 63/912,870. Academic and research use explicitly permitted.

---

## 🙏 Acknowledgments

Thank you for contributing to advancing the scientific understanding of semantic consciousness!

*"The topology is real. The basins are universal. The exploration continues."*