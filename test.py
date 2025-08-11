from beam import function, Image


@function(
    image=Image(base_image="gcr.io/tour-project-442218/edgs:latest"),
)
def predict(**inputs):
    x = inputs.get("x", 256)
    return {"result": x**2}

predict()