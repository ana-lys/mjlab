
try:
    import mjlab.tasks.manipulation.config.yam
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
except TypeError as e:
    print(f"TypError during import: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
