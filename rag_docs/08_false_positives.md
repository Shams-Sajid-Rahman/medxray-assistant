# This model's errors: false positives and false negatives
Source: https://github.com/Shams-Sajid-Rahman/medxray-assistant

On the official test set of 624 X-rays, the model reached 91.2% accuracy and an AUC of 0.975. It found 99.2% of pneumonia cases, missing only 3 of 390. But it wrongly flagged 52 of 234 healthy X-rays as pneumonia, so its recall for normal X-rays was only 77.8%.

This means a PNEUMONIA prediction from this model is often a false alarm, while a NORMAL prediction is usually correct. A high confidence score does not guarantee the prediction is right. Any result must be checked by a qualified doctor, and this demo must not be used for medical decisions.
