import jax.numpy as jnp

def add(x):
    return jnp.sum(x)

def main():
    a = jnp.linspace(1, 10, 5)
    print("Hello from tnt!", a)


if __name__ == "__main__":
    main()
