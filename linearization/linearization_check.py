def test_constraints(test_cases, non_linear_constraint, linear_constraints):
    """
    Test if linear constraints accurately reflect the non-linear constraint over a set of test cases.

    Parameters:
    - test_cases (list of dict): A list of test cases, where each test case is a dictionary mapping variable names to values.
    - non_linear_constraint (function): A function that takes a dictionary of variables and returns True if the non-linear constraint is satisfied, False otherwise.
    - linear_constraints (list of functions): A list of functions that take a dictionary of variables and return True if each linear constraint is satisfied, False otherwise.

    Returns:
    - results (list of dict): A list of results for each test case, showing whether the non-linear and linear constraints match.
    """

    results = []

    for i, test_case in enumerate(test_cases):
        # Check the non-linear constraint
        non_linear_result = non_linear_constraint(test_case)

        # Check all linear constraints
        linear_result = all(constraint(test_case) for constraint in linear_constraints)

        # Store the result for this test case
        match = non_linear_result == linear_result
        result = {
            'test_case': test_case,
            'non_linear_result': non_linear_result,
            'linear_result': linear_result,
            'match': match
        }
        results.append(result)

        # Print summary for this test case
        print(f"Test Case {i + 1}:")
        print(f"  Variables: {test_case}")
        print(f"  Non-linear constraint: {'Passed' if non_linear_result else 'Failed'}")
        print(f"  Linear constraints: {'Passed' if linear_result else 'Failed'}")
        print(f"  Match: {'Yes' if match else 'No'}")
        print("-" * 40)

    return results


# Example of how you might define the non-linear constraint and linear constraints

# Example non-linear constraint: U + min(x, y) < z
def non_linear_constraint(case):
    U, x, y, z = case['U'], case['x'], case['y'], case['z']
    return U + min(x, y) < z


# Example linear constraints: w <= x, w <= y, U + w <= z - epsilon
def linear_constraint_1(case):
    return case['w'] <= case['x']


def linear_constraint_2(case):
    return case['w'] <= case['y']


def linear_constraint_3(case):
    epsilon = 0.0001  # small epsilon for strict inequality
    return case['U'] + case['w'] <= case['z'] - epsilon


# Example test cases
test_cases = [
    {'U': 2, 'x': 5, 'y': 3, 'z': 6, 'w': 3},
    {'U': 1, 'x': 8, 'y': 7, 'z': 9, 'w': 7},
    {'U': 3, 'x': 4, 'y': 5, 'z': 8, 'w': 4},
    {'U': 2, 'x': 7, 'y': 7, 'z': 10, 'w': 7},
]

# Define the linear constraints as a list of functions
linear_constraints = [linear_constraint_1, linear_constraint_2, linear_constraint_3]

# Run the test
results = test_constraints(test_cases, non_linear_constraint, linear_constraints)
