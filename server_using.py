from server import generate_image
a = generate_image.delay("generate")
print(a.ready())
print(a.get())