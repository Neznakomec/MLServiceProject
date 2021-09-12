from tasks import add
a = add.delay(4, 4)
print(a.ready())
print(a.get())