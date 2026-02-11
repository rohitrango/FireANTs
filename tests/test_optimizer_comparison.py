"""
Compare final Dice scores for deformable registration with Adam vs Levenberg
optimizers, for both Greedy and SyN methods.
"""
import logging
import pytest
from pathlib import Path

from fireants.registration.greedy import GreedyRegistration
from fireants.registration.syn import SyNRegistration
from fireants.io.image import Image, BatchedImages

try:
    from .conftest import dice_loss
except ImportError:
    from conftest import dice_loss

logger = logging.getLogger(__name__)

test_data_dir = Path(__file__).parent / "test_data"
fixed_image_path = str(test_data_dir / "deformable_image_1.nii.gz")
moving_image_path = str(test_data_dir / "deformable_image_2.nii.gz")
fixed_seg_path = str(test_data_dir / "deformable_seg_1.nii.gz")
moving_seg_path = str(test_data_dir / "deformable_seg_2.nii.gz")


def _run_registration_and_dice(method: str, optimizer: str):
    """Run registration with given method and optimizer; return mean Dice score."""
    fixed_image = Image.load_file(fixed_image_path)
    moving_image = Image.load_file(moving_image_path)
    fixed_seg = Image.load_file(fixed_seg_path, is_segmentation=True)
    moving_seg = Image.load_file(moving_seg_path, is_segmentation=True)
    fixed_batch = BatchedImages([fixed_image])
    moving_batch = BatchedImages([moving_image])
    fixed_seg_batch = BatchedImages([fixed_seg])
    moving_seg_batch = BatchedImages([moving_seg])

    common = dict(
        scales=[4, 2, 1],
        iterations=[200, 100, 50],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type="fusedcc",
        optimizer=optimizer,
        optimizer_lr=0.5,
        smooth_warp_sigma=0.25,
        smooth_grad_sigma=0.5,
    )

    if method == "greedy":
        reg = GreedyRegistration(**common)
    else:
        reg = SyNRegistration(**common)

    reg.optimize()
    moved_seg_batch = reg.evaluate(fixed_seg_batch, moving_seg_batch).detach()
    dice_scores = (
        1
        - dice_loss(
            moved_seg_batch,
            fixed_seg_batch(),
            reduce=False,
        ).mean(0)
    )
    return float(dice_scores.mean()), dice_scores


def test_final_dice_score_greedy_adam():
    """Final Dice for Greedy registration with Adam optimizer."""
    mean_dice, dice_scores = _run_registration_and_dice("greedy", "adam")
    logger.info("Greedy + Adam:")
    logger.info(f"  Average Dice score: {mean_dice:.3f}")
    logger.info(f"  Labels with Dice > 0.7: {sum(1 for d in dice_scores if d > 0.7)}")
    logger.info(f"  Labels with Dice > 0.8: {sum(1 for d in dice_scores if d > 0.8)}")
    logger.info(f"  Labels with Dice > 0.9: {sum(1 for d in dice_scores if d > 0.9)}")
    assert mean_dice > 0.75, f"Average Dice score ({mean_dice:.3f}) is below threshold"


def test_final_dice_score_greedy_levenberg():
    """Final Dice for Greedy registration with Levenberg optimizer."""
    mean_dice, dice_scores = _run_registration_and_dice("greedy", "levenberg")
    logger.info("Greedy + Levenberg:")
    logger.info(f"  Average Dice score: {mean_dice:.3f}")
    logger.info(f"  Labels with Dice > 0.7: {sum(1 for d in dice_scores if d > 0.7)}")
    logger.info(f"  Labels with Dice > 0.8: {sum(1 for d in dice_scores if d > 0.8)}")
    logger.info(f"  Labels with Dice > 0.9: {sum(1 for d in dice_scores if d > 0.9)}")
    assert mean_dice > 0.75, f"Average Dice score ({mean_dice:.3f}) is below threshold"


def test_final_dice_score_syn_adam():
    """Final Dice for SyN registration with Adam optimizer."""
    mean_dice, dice_scores = _run_registration_and_dice("syn", "adam")
    logger.info("SyN + Adam:")
    logger.info(f"  Average Dice score: {mean_dice:.3f}")
    logger.info(f"  Labels with Dice > 0.7: {sum(1 for d in dice_scores if d > 0.7)}")
    logger.info(f"  Labels with Dice > 0.8: {sum(1 for d in dice_scores if d > 0.8)}")
    logger.info(f"  Labels with Dice > 0.9: {sum(1 for d in dice_scores if d > 0.9)}")
    assert mean_dice > 0.75, f"Average Dice score ({mean_dice:.3f}) is below threshold"


def test_final_dice_score_syn_levenberg():
    """Final Dice for SyN registration with Levenberg optimizer."""
    mean_dice, dice_scores = _run_registration_and_dice("syn", "levenberg")
    logger.info("SyN + Levenberg:")
    logger.info(f"  Average Dice score: {mean_dice:.3f}")
    logger.info(f"  Labels with Dice > 0.7: {sum(1 for d in dice_scores if d > 0.7)}")
    logger.info(f"  Labels with Dice > 0.8: {sum(1 for d in dice_scores if d > 0.8)}")
    logger.info(f"  Labels with Dice > 0.9: {sum(1 for d in dice_scores if d > 0.9)}")
    assert mean_dice > 0.75, f"Average Dice score ({mean_dice:.3f}) is below threshold"
