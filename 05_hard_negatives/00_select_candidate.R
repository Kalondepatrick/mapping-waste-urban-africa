Here we identify waste piles that were missed by the model (i.e. lowe probability than other class - the one assigned by the sigmoid function)
and those that are false positives. We randomly select a few of these for manual digitization and the model is fine-tuned
This has to be tested. So digitize some test location and know number of waste piles, check how many of those were detected at the end