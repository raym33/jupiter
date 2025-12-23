# Creating a New Domain

This guide explains how to configure a new specialization domain for Jupiter.

## What is a Domain?

A domain defines **what type of expert** you want to train. It includes:

- **Data sources**: Where to get information (docs, code, web)
- **Generation templates**: How to create synthetic data
- **Evaluation criteria**: How to measure model quality

## Step 1: Create the Domain File

```bash
jupiter new-domain my_domain
```

This creates `config/domains/my_domain.yaml` from the template.

## Step 2: Configure the Domain

Edit the YAML file:

### Basic Information

```yaml
domain:
  name: "python_expert"
  description: "Expert in Python programming"
  language: "en"  # or "es"

  # Keywords to filter relevant content
  keywords:
    - "python"
    - "pip"
    - "django"
    - "flask"
    - "pandas"
    - "numpy"

  # Content to avoid
  negative_keywords:
    - "javascript"  # Different language
    - "java"
```

### Data Sources

```yaml
data_sources:
  # Official documentation
  documentation:
    - "https://docs.python.org/3/"
    - "https://docs.djangoproject.com/"

  # GitHub code
  github:
    repos:
      - "python/cpython"
      - "django/django"
      - "pallets/flask"
    file_types:
      - ".py"
      - ".md"
      - ".rst"
    exclude_patterns:
      - "test_*"
      - "*_test.py"
      - "docs/conf.py"

  # Websites
  websites:
    - "https://realpython.com"
    - "https://www.fullstackpython.com"

  # Forums
  forums:
    - "https://discuss.python.org"
```

### Generation Templates

Templates define how the LLM will generate synthetic data:

```yaml
generation:
  templates:
    # Questions and Answers
    - type: "qa"
      system_prompt: |
        You are a Python expert with 15 years of experience.
        You answer clearly with code examples.
      prompt: |
        Generate a technical question about {topic} in Python
        and its detailed answer.

        Include:
        1. Concept explanation
        2. Code example
        3. Common errors
        4. Best practices
      topics:
        - "decorators"
        - "generators"
        - "context managers"
        - "asyncio"
        - "type hints"
        - "dataclasses"

    # Tutorials
    - type: "tutorial"
      system_prompt: |
        You are a Python instructor.
      prompt: |
        Create a step-by-step tutorial for {task}.
      tasks:
        - "create a REST API with FastAPI"
        - "web scraping with BeautifulSoup"
        - "data analysis with pandas"

    # Debugging
    - type: "debug"
      prompt: |
        Generate a common Python error related to {topic}.
        Include the traceback and solution.
      topics:
        - "IndentationError"
        - "ImportError"
        - "AttributeError"
        - "TypeError"

  # Proportion of each type
  template_weights:
    qa: 0.40
    tutorial: 0.35
    debug: 0.25
```

### Evaluation

```yaml
evaluation:
  benchmarks:
    - "code_syntax"      # Is the code valid?
    - "best_practices"   # Follows PEP8?
    - "completeness"     # Is the answer complete?

  min_accuracy: 0.75
  min_coherence: 0.80
```

### Data Mix Ratio

```yaml
mix_ratio:
  real_docs: 0.30        # 30% real documentation
  real_code: 0.25        # 25% real code
  synthetic_qa: 0.20     # 20% generated Q&A
  synthetic_tutorials: 0.15
  synthetic_debug: 0.10
```

## Domain Examples

### University Chemistry

```yaml
domain:
  name: "chemistry"
  description: "Expert in university chemistry"
  language: "en"

  keywords:
    - "atom"
    - "molecule"
    - "bond"
    - "reaction"
    - "equilibrium"
    - "thermodynamics"

data_sources:
  documentation:
    - "https://chem.libretexts.org"

  websites:
    - "https://www.khanacademy.org/science/chemistry"

generation:
  templates:
    - type: "qa"
      prompt: |
        Generate a university exam question about {topic}
        with its detailed answer and step-by-step explanation.
      topics:
        - "stoichiometry"
        - "chemical equilibrium"
        - "kinetics"
        - "thermodynamics"
        - "electrochemistry"

    - type: "explanation"
      prompt: |
        Explain the concept of {topic} as if you were a professor.
        Include examples and analogies.
      topics:
        - "atomic orbitals"
        - "hybridization"
        - "resonance"
```

### Medicine (Radiology)

```yaml
domain:
  name: "radiology"
  description: "Expert in radiological interpretation"
  language: "en"

  keywords:
    - "x-ray"
    - "CT scan"
    - "MRI"
    - "ultrasound"
    - "diagnosis"

generation:
  templates:
    - type: "qa"
      prompt: |
        Describe the typical radiological findings of {condition}
        and the differential diagnosis.
      topics:
        - "pneumonia"
        - "hip fracture"
        - "brain tumor"
        - "pleural effusion"
```

## Step 3: Test the Domain

```bash
# Verify the domain loads correctly
jupiter check

# Run a quick test
jupiter start --domain my_domain --epochs 1 --no-collect
```

## Tips

1. **Start simple**: Few topics, few sources
2. **Iterate**: Add more sources and templates gradually
3. **Review quality**: Synthetic data should be useful
4. **Balance**: Maintain real vs synthetic data ratio

## Next Steps

- [Set up Mac cluster](mac_cluster_setup.md)
- [Understand self-improvement](self_improvement.md)
