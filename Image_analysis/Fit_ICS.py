import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
PIXEL_SIZE = 0.05


def ICS_fit(normalized_autocorrelation: np.ndarray, ParamIn: Optional[Dict[str, List[float]]] = None, method: str = 'Nelder-Mead', max_evaluations: int = 600) -> Tuple[np.ndarray, List[float], Dict[str, Any]]:
    # logger.info("Starting the fitting process...")
    AC = normalized_autocorrelation
    if ParamIn is None:
        ParamIn = {
            'varie': [1, 1, 1],
            'val': [1/(np.max(AC)-1), 0.25, 0]
        }
        # logger.info(f"Using default parameters: {ParamIn}")
    
    is_1d = AC.ndim == 1
    x, y = _prepare_coordinates(AC, is_1d)
    
    def chi(p: List[float]) -> float:
        return np.sum((AC - _calculate_G(p, x, y, is_1d, ParamIn))**2)
    
    start = ParamIn['val'][:3] if ParamIn['varie'] == [1, 1, 1] else [ParamIn['val'][0]]
    # logger.info(f"Starting optimization with initial guess: {start}")
    
    result = None
    for attempt in range(2):  # Try twice: once with default, once with increased evaluations
        try:
            result = minimize(chi, start, method=method, tol=1E-8, options={'maxfev': max_evaluations})
            # logger.info("Optimization completed.")
            if result.success or result.status != 1:  # If successful or failed for reasons other than max evaluations
                break
            else:
                logger.warning(f"Optimization reached max evaluations ({max_evaluations}). Retrying with increased limit.")
                max_evaluations *= 1  # Double the max evaluations for the next attempt
        except Exception as e:
            logger.error(f"Optimization failed with error: {e}")
            if attempt == 1:  # If this was the second attempt, raise the exception
                raise
    
    if result is None:
        # logger.critical("Fitting failed in all attempts.")
        return np.full_like(AC, np.nan), [np.nan, np.nan, np.nan], {'success': False, 'message': 'All fitting attempts failed'}

    pf = result.x
    # logger.info(f"Optimization results: {pf}")
    
    debug = {
        'success': result.success,
        'status': result.status,
        'message': result.message,
        'nfev': result.nfev,
        'nit': result.nit
    }
    
    if not debug['success']:
        logger.critical("Fitting was not successful. Returning NaN values.")
        logger.critical(f"Debug information: {debug}")
        Fit = np.full_like(AC, np.nan)
        ParamOut = [np.nan, np.nan, np.nan]
    else:
        ParamOut = list(pf) if ParamIn['varie'] == [1, 1, 1] else [pf[0], ParamIn['val'][1], ParamIn['val'][2]]
        Fit = _calculate_G(ParamOut, x, y, is_1d, ParamIn)
    
    # logger.info("Fitting completed. Results: " + ("Successful" if result.success else "Not successful"))
    # logger.info(f"Fitting debug information: {debug}")
    
    return Fit, ParamOut, debug

@staticmethod
def _prepare_coordinates(AC: np.ndarray, is_1d: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if is_1d:
        return np.arange(len(AC)), None
    else:
        size = AC.shape[0]
        y, x = np.indices(AC.shape)
        return x - size // 2, y - size // 2

def _calculate_G(p: List[float], x: np.ndarray, y: Optional[np.ndarray], is_1d: bool, ParamIn: Dict[str, List[float]]) -> np.ndarray:
    if is_1d:
        return 1 / p[0] * np.exp(-(x * PIXEL_SIZE)**2 / (p[1] if len(p) == 3 else ParamIn['val'][1])**2) + 1 + (p[2] if len(p) == 3 else ParamIn['val'][2])
    else:
        return 1 / p[0] * np.exp(-((x * PIXEL_SIZE)**2 + (y * PIXEL_SIZE)**2) / (p[1] if len(p) == 3 else ParamIn['val'][1])**2) + 1 + (p[2] if len(p) == 3 else ParamIn['val'][2])
