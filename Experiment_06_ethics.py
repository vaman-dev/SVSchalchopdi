"""Experiment 06: practical demonstrations of Ethics and Responsible AI."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_SEED = 42
MODEL_FEATURES = ["qualification_score", "experience_years"]
GROUP_COLUMN = "audit_group"
TARGET_COLUMN = "selected"


def create_grouped_dataset():
    """Create an intentionally imbalanced dataset for an educational audit."""
    rng = np.random.default_rng(RANDOM_SEED)
    group_settings = {
        "Group A": {"size": 400, "noise": 0.60},
        "Group B": {"size": 100, "noise": 1.35},
    }
    frames = []
    sample_number = 0

    for group_name, settings in group_settings.items():
        size = settings["size"]
        qualification = np.clip(rng.normal(70, 12, size), 30, 100)
        experience = np.clip(rng.normal(5, 2.5, size), 0, 15)

        # Both groups use the same underlying decision rule. Group B receives
        # more label noise so the group-wise audit has a meaningful difference
        # to investigate. This is a synthetic teaching example, not real data.
        decision_score = (
            0.08 * (qualification - 70)
            + 0.25 * (experience - 5)
            + rng.normal(0, settings["noise"], size)
        )
        selected = (decision_score >= 0).astype(int)

        frames.append(
            pd.DataFrame(
                {
                    "sample_id": np.arange(sample_number, sample_number + size),
                    GROUP_COLUMN: group_name,
                    "qualification_score": qualification,
                    "experience_years": experience,
                    TARGET_COLUMN: selected,
                }
            )
        )
        sample_number += size

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    assert set(dataset.columns) == {
        "sample_id",
        GROUP_COLUMN,
        *MODEL_FEATURES,
        TARGET_COLUMN,
    }
    assert dataset["sample_id"].is_unique
    return dataset


def analyze_representation(dataset):
    """Calculate and plot the representation of each audit group."""
    counts = dataset[GROUP_COLUMN].value_counts().sort_index()
    percentages = counts.div(len(dataset)).mul(100)

    assert counts.to_dict() == {"Group A": 400, "Group B": 100}
    assert np.isclose(percentages.sum(), 100.0)

    representation = pd.DataFrame(
        {"sample_count": counts, "representation_percent": percentages}
    )
    underrepresented_group = counts.idxmin()

    print("\nPART A - DATASET REPRESENTATION")
    print(representation.to_string(float_format=lambda value: f"{value:.2f}"))
    print("Underrepresented group:", underrepresented_group)
    print(
        "Interpretation: unequal representation is a bias risk that should be "
        "investigated; it does not alone prove discriminatory outcomes."
    )

    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(counts.index, counts.values, color=["#4C78A8", "#F58518"])
    axis.bar_label(bars, labels=[str(value) for value in counts.values])
    axis.set_title("Dataset Representation by Group")
    axis.set_xlabel("Audit group")
    axis.set_ylabel("Number of samples")
    axis.set_ylim(0, counts.max() * 1.15)
    figure.tight_layout()
    plt.show()

    return representation


def evaluate_group_fairness(dataset):
    """Train without the audit attribute and report held-out group accuracy."""
    stratification_key = (
        dataset[GROUP_COLUMN] + "_" + dataset[TARGET_COLUMN].astype(str)
    )
    train_data, test_data = train_test_split(
        dataset,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=stratification_key,
    )

    assert set(train_data["sample_id"]).isdisjoint(test_data["sample_id"])
    assert GROUP_COLUMN not in MODEL_FEATURES
    assert TARGET_COLUMN not in MODEL_FEATURES

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
    )
    model.fit(train_data[MODEL_FEATURES], train_data[TARGET_COLUMN])
    test_data = test_data.copy()
    test_data["prediction"] = model.predict(test_data[MODEL_FEATURES])

    group_accuracies = test_data.groupby(GROUP_COLUMN, sort=True).apply(
        lambda group: accuracy_score(group[TARGET_COLUMN], group["prediction"]),
        include_groups=False,
    )
    accuracy_gap = float(group_accuracies.max() - group_accuracies.min())

    assert set(group_accuracies.index) == {"Group A", "Group B"}
    assert group_accuracies.between(0.0, 1.0).all()
    assert 0.0 <= accuracy_gap <= 1.0

    print("\nPART B - GROUP FAIRNESS EVALUATION")
    print("Model features:", ", ".join(MODEL_FEATURES))
    print("Audit-only attribute excluded from model:", GROUP_COLUMN)
    print("Training rows:", len(train_data))
    print("Held-out test rows:", len(test_data))
    for group_name, group_accuracy in group_accuracies.items():
        group_size = int((test_data[GROUP_COLUMN] == group_name).sum())
        print(f"{group_name} test accuracy: {group_accuracy:.4f} (n={group_size})")
    print(f"Absolute group accuracy gap: {accuracy_gap:.4f}")
    print(
        "Interpretation: this gap is a screening signal, not proof of fairness "
        "or unfairness. Sample size, label quality, error types, and application "
        "context must also be examined."
    )

    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(
        group_accuracies.index,
        group_accuracies.values,
        color=["#4C78A8", "#F58518"],
    )
    axis.bar_label(
        bars,
        labels=[f"{value:.1%}" for value in group_accuracies.values],
    )
    axis.set_title("Held-out Accuracy by Audit Group")
    axis.set_xlabel("Audit group")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1.1)
    figure.tight_layout()
    plt.show()

    return model, group_accuracies, accuracy_gap


def explain_iris_model():
    """Train an Iris Decision Tree and visualize its feature importance."""
    iris = load_iris(as_frame=True)
    features = iris.data
    targets = iris.target
    train_features, test_features, train_targets, test_targets = train_test_split(
        features,
        targets,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=targets,
    )

    model = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED)
    model.fit(train_features, train_targets)
    predictions = model.predict(test_features)
    test_accuracy = accuracy_score(test_targets, predictions)
    feature_importance = pd.Series(
        model.feature_importances_,
        index=features.columns,
        name="importance",
    ).sort_values(ascending=False)

    assert 0.0 <= test_accuracy <= 1.0
    assert (feature_importance >= 0).all()
    assert np.isclose(feature_importance.sum(), 1.0)

    print("\nPART C - EXPLAINABLE AI WITH IRIS")
    print(f"Decision Tree held-out accuracy: {test_accuracy:.4f}")
    print("Feature importance:")
    print(feature_importance.to_string(float_format=lambda value: f"{value:.4f}"))
    print(
        "Interpretation: impurity-based importance describes how this tree used "
        "the features. It does not establish causation."
    )

    figure, axis = plt.subplots(figsize=(9, 4))
    bars = axis.bar(
        feature_importance.index,
        feature_importance.values,
        color="#54A24B",
    )
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in feature_importance])
    axis.set_title("Iris Decision Tree Feature Importance")
    axis.set_xlabel("Feature")
    axis.set_ylabel("Relative importance")
    axis.tick_params(axis="x", rotation=20)
    axis.set_ylim(0, max(feature_importance.max() * 1.15, 0.1))
    figure.tight_layout()
    plt.show()

    return model, test_accuracy, feature_importance


def print_responsible_ai_case_study():
    """Print a structured ethical analysis of an AI recruitment system."""
    considerations = {
        "Bias": "Historical hiring data may reproduce past preferences or exclusions.",
        "Fairness": "Compare error and selection metrics across relevant groups.",
        "Privacy": "Collect only necessary applicant data and protect access to it.",
        "Safety": "Incorrect rejection can materially affect a person's opportunity.",
        "Explainability": "Applicants and reviewers need understandable reasons.",
        "Accountability": "An identified organization and owner must answer for outcomes.",
        "Human involvement": "Humans should review edge cases and support appeals.",
    }

    print("\nPART D - RESPONSIBLE AI CASE STUDY: AI RECRUITMENT")
    for topic, discussion in considerations.items():
        print(f"- {topic}: {discussion}")
    print(
        "Recommended controls: document data sources, test subgroup performance, "
        "monitor drift, restrict sensitive data, record decisions, and maintain "
        "meaningful human review and an appeal process."
    )


def print_autonomous_system_analysis():
    """Print essential governance questions for autonomous systems."""
    considerations = {
        "Safety": "Use fail-safe behavior, hazard analysis, and real-world testing.",
        "Reliability": "Test unusual conditions, degraded sensors, and component failures.",
        "Accountability": "Define responsibility across developers, owners, and operators.",
        "Privacy": "Limit collection and retention of location, video, and personal data.",
        "Security": "Protect models, sensors, communications, and update mechanisms.",
        "Human control": "Provide monitoring, override, safe shutdown, and escalation paths.",
    }

    print("\nPART E - AUTONOMOUS SYSTEMS")
    for topic, discussion in considerations.items():
        print(f"- {topic}: {discussion}")


def main():
    """Run all five parts of the Responsible AI experiment."""
    print("EXPERIMENT 06 - ETHICS AND RESPONSIBLE AI")
    dataset = create_grouped_dataset()
    print("Synthetic dataset shape:", dataset.shape)

    analyze_representation(dataset)
    evaluate_group_fairness(dataset)
    explain_iris_model()
    print_responsible_ai_case_study()
    print_autonomous_system_analysis()

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
