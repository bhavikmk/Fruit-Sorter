# Fruit-Sorter (Mango)

## Abstract

The Mango Sorting Machine is a solution designed to automate the process of sorting and grading mangoes based on three critical factors: size, extent of ripeness, and spoilage. Traditional sorting and grading methods rely heavily on manual labor, which is often inefficient, time-consuming, and prone to inconsistencies. This project introduces a computer vision-based system that integrates automatic grading with systematic sorting, enhancing the overall efficiency and precision of mango processing.

### Automatic grading 

Grading refers to the classification of mangoes based on specific features. The traditional approach involves skilled workers manually inspecting the fruit, which is labor-intensive and inconsistent. Automation addresses these challenges by accurately analyzing the following parameters:

**Size & Shape:**

- Mango dimensions (e.g., length, width) are measured using image processing techniques.
- Shape analysis ensures mangoes meet uniformity standards for packaging and sale.

**Ripeness Evaluation:**

- Ripeness is determined through color analysis, where color variations indicate different stages of maturity (e.g., unripe, partially ripe, ripe).
- Wrinkle detection uses texture mapping to evaluate surface characteristics, a key indicator of over-ripeness.

**Spoilage Detection:**

- Spoiled areas are identified by detecting changes in color and texture.
- The extent of spoilage is quantified to classify mangoes into acceptable or rejected categories.

### Automatic Sorting

Sorting involves arranging mangoes systematically into predefined categories based on grading results. This step is crucial for efficient packaging and further processing. The system uses conveyor belts and sorting mechanisms (e.g., robotic arms or pneumatic systems) to direct mangoes into designated bins or packaging units.


Sorting Criteria:

- **Size**: Small, medium, or large.
- **Maturity**: Unripe, early ripe, partially ripe, or ripe.
- **Spoilage**: Accepted or rejected based on spoilage percentage.

### Performance Goals:

- Real-time processing with at least 30 frames per second (FPS).

**Classification accuracy**:
- Size & Shape: >95%.
- Spoilage: >90%.
- Maturity: >93%.
- Sorting throughput: Capable of processing 100 mangoes per minute.

### Applications

- **Agricultural Co-Operatives**: Streamlines grading and sorting operations for mango farmers.
- **Food Processing Units**: Ensures quality control in mango-based product manufacturing.
- **Export Industry**: Meets international standards for mango quality by providing precise grading and sorting.

## References

1. Naik, Sapan, Bankim Patel, and Rashmi Pandey. "Shape, size and maturity features extraction with fuzzy classifier for non-destructive mango (Mangifera Indica L., cv. Kesar) grading." 2015 IEEE Technological Innovation in ICT for Agriculture and Rural Development (TIAR). IEEE, 2015.
2. [Dataset for Ripeness](https://www.kaggle.com/datasets/mutiurrehman80/ripeness-detection-of-mango)
