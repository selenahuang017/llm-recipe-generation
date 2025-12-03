#pip install sacrebleu
from typing import Dict, List, Optional
from collections import defaultdict

try:
    from sacrebleu import BLEU
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: sacrebleu not installed. BLEU metrics will not be available.")


class MealPlanMetrics:
    """
    Metrics collection and calculation for meal plan generation.
    Tracks validation success, iteration counts, and constraint satisfaction.
    """
    
    def __init__(self):
        self.validation_results: List[Dict] = []
        self.iteration_counts: List[int] = []
        self.constraint_satisfaction: List[Dict] = []
        self.meal_plans: List[str] = []
        self.iteration_history: List[List[str]] = []  # Store meal plans at each iteration
        
    def record_run(self, 
                   validation_result: Dict[str, any],
                   iterations: int,
                   meal_plan: str,
                   iteration_history: Optional[List[str]] = None):
        """
        Record a single meal plan generation run.
        
        Args:
            validation_result: Dict with 'success' and 'message' from validator
            iterations: Number of iterations taken (1-indexed, so 1 means passed on first try)
            meal_plan: Final meal plan string
            iteration_history: Optional list of meal plans at each iteration (for self-BLEU)
        """
        self.validation_results.append(validation_result)
        self.iteration_counts.append(iterations)
        self.meal_plans.append(meal_plan)
        if iteration_history:
            self.iteration_history.append(iteration_history)
        
        # Extract constraint satisfaction details from validation result
        constraint_info = self._extract_constraint_info(validation_result)
        self.constraint_satisfaction.append(constraint_info)
    
    def _extract_constraint_info(self, validation_result: Dict) -> Dict[str, bool]:
        """
        Extract individual constraint satisfaction from validation message.
        """
        message = validation_result.get('message', '').lower()
        
        return {
            'structure': 'meal plan structure' in message and 'correct' in message,
            'ingredients': 'ingredients' in message and ('used' in message or 'present' in message),
            'allergens': 'allergen' not in message or ('no allergens' in message or 'allergens specified' in message),
            'calories': 'calories' in message and ('within limits' in message or 'exceeds' not in message),
            'budget': 'budget' not in message or ('within budget' in message or 'no budget specified' in message),
            'sections': 'sections' in message and 'present' in message
        }
    
    def validation_success_rate(self) -> float:
        """
        Calculate the percentage of meal plans that passed all validation checks.
        
        Returns:
            Float between 0.0 and 1.0 representing success rate
        """
        if not self.validation_results:
            return 0.0
        
        successful = sum(1 for r in self.validation_results if r.get('success', False))
        return successful / len(self.validation_results)
    
    def average_iterations(self) -> float:
        """
        Calculate average number of iterations needed to pass validation.
        Only counts successful validations.
        
        Returns:
            Average iterations (float), or 0.0 if no successful runs
        """
        successful_iterations = [
            self.iteration_counts[i] 
            for i, result in enumerate(self.validation_results)
            if result.get('success', False)
        ]
        
        if not successful_iterations:
            return 0.0
        
        return sum(successful_iterations) / len(successful_iterations)
    
    def constraint_satisfaction_rate(self) -> Dict[str, float]:
        """
        Calculate satisfaction rate for each constraint type.
        
        Returns:
            Dict mapping constraint names to satisfaction rates (0.0-1.0)
        """
        if not self.constraint_satisfaction:
            return {}
        
        constraint_counts = defaultdict(int)
        total_runs = len(self.constraint_satisfaction)
        
        for constraints in self.constraint_satisfaction:
            for constraint, satisfied in constraints.items():
                if satisfied:
                    constraint_counts[constraint] += 1
        
        return {
            constraint: count / total_runs
            for constraint, count in constraint_counts.items()
        }
    
    def get_metrics_summary(self) -> Dict[str, any]:
        """
        Get a comprehensive summary of all metrics.
        
        Returns:
            Dict with all metric values
        """
        return {
            'validation_success_rate': self.validation_success_rate(),
            'average_iterations': self.average_iterations(),
            'total_runs': len(self.validation_results),
            'successful_runs': sum(1 for r in self.validation_results if r.get('success', False)),
            'constraint_satisfaction': self.constraint_satisfaction_rate(),
            'iteration_distribution': self._get_iteration_distribution()
        }
    
    def _get_iteration_distribution(self) -> Dict[int, int]:
        """
        Get distribution of iteration counts.
        
        Returns:
            Dict mapping iteration count to number of runs
        """
        distribution = defaultdict(int)
        for count in self.iteration_counts:
            distribution[count] += 1
        return dict(distribution)


def calculate_self_bleu(iteration_history: List[str]) -> Optional[float]:
    """
    Calculate BLEU score by comparing first iteration to last iteration.
    Measures how much the meal plan changed during validation iterations.
    
    Args:
        iteration_history: List of meal plans at each iteration (first to last)
    
    Returns:
        BLEU score (0.0-1.0) or None if BLEU not available or insufficient data
    """
    if not BLEU_AVAILABLE:
        return None
    
    if len(iteration_history) < 2:
        return None
    
    # Compare first iteration to last iteration
    first_iteration = iteration_history[0]
    last_iteration = iteration_history[-1]
    
    try:
        bleu = BLEU()
        score = bleu.corpus_score([last_iteration], [[first_iteration]])
        return score.score / 100.0  # Convert to 0-1 scale
    except Exception:
        return None


# Convenience function for backward compatibility
def test_bleu_score(recipe: str, reference: Optional[str] = None) -> Optional[float]:
    """
    Calculate BLEU score for a recipe.
    If reference provided, compares against it. Otherwise returns None.
    
    Args:
        recipe: Generated recipe/meal plan
        reference: Optional reference recipe/meal plan
    
    Returns:
        BLEU score (0.0-1.0) or None
    """
    if not BLEU_AVAILABLE:
        return None
    
    if reference is None:
        return None
    
    try:
        bleu = BLEU()
        score = bleu.corpus_score([recipe], [[reference]])
        return score.score / 100.0
    except Exception:
        return None
