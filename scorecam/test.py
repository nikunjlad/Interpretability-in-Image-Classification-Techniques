import click


class Test:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return "Name: {}, Age: {}".format(self.name, str(self.age))


@click.command()
@click.option("-n", "--name", type=str, required=True)
@click.option("-a", "--age", type=int, default=3)
def main(name, age):
    t = Test(name, age)
    print(t)


if __name__ == "__main__":
    main()