'''
Copyright (c) 2019 Keiron O'Shea

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public
License along with this program; if not, write to the
Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
Boston, MA 02110-1301 USA
'''

from keras.callbacks import (
    EarlyStopping,
    History,
    ModelCheckpoint,
)


class Classifier(object):
    def __init__(self, height, width, max_frames, clf):
        self.height = height
        self.width = width
        self.max_frames = max_frames
        self.trained = False
        self.clf = clf

    def fit(self,
            generator,
            validation_data,
            model_fp: str,
            class_weights=False,
            epochs: int = 10,
            patience: int = 5) -> None:

        if self.trained:
            raise RuntimeError("The model has already been trained")

        hi = History()

        mc = ModelCheckpoint(
            model_fp, monitor="val_loss", verbose=1, save_best_only=True)

        es = EarlyStopping(monitor="val_loss", verbose=1, patience=patience)

        clf = self.clf

        if class_weights:
            history = clf.fit_generator(
                generator,
                epochs=epochs,
                callbacks=[hi, mc, es],
                validation_data=validation_data,
                verbose=1,
                class_weight=class_weights
                )
        else:
            history = clf.fit_generator(
                generator,
                epochs=epochs,
                callbacks=[hi, mc, es],
                validation_data=validation_data,
                verbose=1
                )

        self.clf.load_weights(model_fp)
        self.history = history
        self.trained = True

    def predict_pod(self, generator):
        y_true = [y for _, y in generator]
        
        y_true = ["Yes" if (x == 1) else "No" for x in y_true]
        y_pred = self.clf.predict_generator(generator)

        for index, pred in enumerate(y_pred):
            if pred >= 0.75:
                y_pred[index] = "Yes"
            elif pred <= 0.35:
                y_pred[index] = "No"
            else:
                y_pred[index] = "Unsure"

        return y_true, y_pred

    def predict_fetal_head_elipsoid(self):
        pass
