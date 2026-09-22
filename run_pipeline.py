import importlib

def main():
    modules = [
        "isl_static_prototype.data_pipeline",
        "isl_static_prototype.training.extract_landmarks",
        "isl_static_prototype.evaluation.audit_landmark_failures",
        "isl_static_prototype.training.build_features",
        "isl_static_prototype.vision.graph_model",
        "isl_static_prototype.training.train",
        "isl_static_prototype.evaluation.evaluate",
    ]

    for module_name in modules:
        importlib.import_module(module_name)

if __name__ == "__main__":
    main()
