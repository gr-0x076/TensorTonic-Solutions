def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Calculates the Wasserstein Critic Loss for WGANs.
    """
    if not real_scores or not fake_scores:
        raise ValueError("Input arrays cannot be empty")
        
    mean_fake = sum(fake_scores) / len(fake_scores)
    mean_real = sum(real_scores) / len(real_scores)
    
    loss = mean_fake - mean_real
    return loss

