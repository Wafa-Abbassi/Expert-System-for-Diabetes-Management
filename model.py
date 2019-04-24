from wtforms import Form, FloatField, validators


class InputForm(Form):
    a = FloatField(
        label='Pregnancies', default=1,
        validators=[validators.InputRequired()])
    b = FloatField(
        label='Glucose', default=1,
        validators=[validators.InputRequired()])
    c = FloatField(
        label='BloodPressure', default=1,
        validators=[validators.InputRequired()])
    d = FloatField(
        label='SkinThickness', default=1,
        validators=[validators.InputRequired()])
    e = FloatField(
        label='Insulin', default=1,
        validators=[validators.InputRequired()])
    z = FloatField(
        label='BMI', default=1,
        validators=[validators.InputRequired()])
    g = FloatField(
        label='DiabetesPedigreeFunction', default=1,
        validators=[validators.InputRequired()])
    h = FloatField(
        label='Age', default=1,
        validators=[validators.InputRequired()])
    