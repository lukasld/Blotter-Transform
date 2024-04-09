
def requiredColumnsNotPresentWarning(actual_names, expected_names):
    """
    Description of the warning
    """
    return f"provided column names: {actual_names}, \
             do not match required names: {expected_names}, this can cause errors later."

def createNewInstanceWarninig(instance_name):
    """
    Description of the warning
    """
    return f"we created a new instance of class {instance_name}"
