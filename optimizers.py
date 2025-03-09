import numpy as np


def SGD(layer: dict, learning_rate: float, config: dict):

    L_theta_by_w_clipped = np.clip(layer["layer"].L_theta_by_w, a_min=-1, a_max=1)
    L_theta_by_b_clipped = np.clip(layer["layer"].L_theta_by_b, a_min=-1, a_max=1)
    layer["layer"].weight -= L_theta_by_w_clipped * learning_rate
    layer["layer"].bias -= L_theta_by_b_clipped * learning_rate


def momentum(layer: dict, learning_rate: float, config: dict):

    layer["layer"].u_w = np.clip(
        (config["momentum_beta"] * layer["layer"].u_w + layer["layer"].L_theta_by_w),
        a_min=-1,
        a_max=1,
    )
    layer["layer"].u_b = np.clip(
        (config["momentum_beta"] * layer["layer"].u_b + layer["layer"].L_theta_by_b),
        a_min=-1,
        a_max=1,
    )

    updated_weight = layer["layer"].u_w * learning_rate
    updated_bias = layer["layer"].u_b * learning_rate

    layer["layer"].weight -= updated_weight
    layer["layer"].bias -= updated_bias


def RMSprop(layer: dict, learning_rate: float, config: dict):
    layer["layer"].v_w = config["RMSprop_beta"] * layer["layer"].v_w + (
        1 - config["RMSprop_beta"]
    ) * np.multiply(layer["layer"].L_theta_by_w, layer["layer"].L_theta_by_w)
    layer["layer"].v_b = config["RMSprop_beta"] * layer["layer"].v_b + (
        1 - config["RMSprop_beta"]
    ) * np.multiply(layer["layer"].L_theta_by_b, layer["layer"].L_theta_by_b)

    updated_weight = np.clip(
        np.multiply(
            layer["layer"].L_theta_by_w,
            (learning_rate / np.sqrt(layer["layer"].v_w + config["RMS_epsilon"])),
        ),
        a_min=-1,
        a_max=1,
    )
    updated_bias = np.clip(
        np.multiply(
            layer["layer"].L_theta_by_b,
            (learning_rate / np.sqrt(layer["layer"].v_b + config["RMS_epsilon"])),
        ),
        a_min=-1,
        a_max=1,
    )

    layer["layer"].weight -= updated_weight
    layer["layer"].bias -= updated_bias


def Adam(layer: dict, learning_rate: float, epoch: int, config: dict):

    ## Setup the Momuntum side of the optimization
    layer["layer"].m_w = (
        config["adam_beta_1"] * layer["layer"].m_w
        + (1 - config["adam_beta_1"]) * layer["layer"].L_theta_by_w
    )
    ## Not adding m_hat to the layer as i dont see the need to track it as of now
    m_hat_w = layer["layer"].m_w / (1 - config["adam_beta_1"] ** epoch)
    layer["layer"].m_b = (
        config["adam_beta_1"] * layer["layer"].m_b
        + (1 - config["adam_beta_1"]) * layer["layer"].L_theta_by_b
    )
    m_hat_b = layer["layer"].m_b / (1 - config["adam_beta_1"] ** epoch)

    ## Setup the V side of the optimization
    layer["layer"].v_w = config["adam_beta_2"] * layer["layer"].v_w + (
        1 - config["adam_beta_2"]
    ) * np.multiply(layer["layer"].L_theta_by_w, layer["layer"].L_theta_by_w)

    v_hat_w = layer["layer"].v_w / (1 - config["adam_beta_2"] ** epoch)

    layer["layer"].v_b = config["adam_beta_2"] * layer["layer"].v_b + (
        1 - config["adam_beta_2"]
    ) * np.multiply(layer["layer"].L_theta_by_b, layer["layer"].L_theta_by_b)

    v_hat_b = layer["layer"].v_b / (1 - config["adam_beta_2"] ** epoch)

    updated_weight = np.clip(
        np.multiply(
            m_hat_w, (learning_rate / (np.sqrt(v_hat_w) + config["RMS_epsilon"]))
        ),
        a_min=-1,
        a_max=1,
    )
    updated_bias = np.clip(
        np.multiply(
            m_hat_b, (learning_rate / (np.sqrt(v_hat_b) + config["RMS_epsilon"]))
        ),
        a_min=-1,
        a_max=1,
    )

    layer["layer"].weight -= updated_weight
    layer["layer"].bias -= updated_bias


def Nadam(layer: dict, learning_rate: float, epoch: int, config: dict):

    ## Setup the Momuntum side of the optimization
    layer["layer"].m_w = (
        config["adam_beta_1"] * layer["layer"].m_w
        + (1 - config["adam_beta_1"]) * layer["layer"].L_theta_by_w
    )
    ## Not adding m_hat to the layer as i dont see the need to track it as of now
    m_hat_w = layer["layer"].m_w / (1 - config["adam_beta_1"] ** (epoch + 1))
    layer["layer"].m_b = (
        config["adam_beta_1"] * layer["layer"].m_b
        + (1 - config["adam_beta_1"]) * layer["layer"].L_theta_by_b
    )
    m_hat_b = layer["layer"].m_b / (1 - config["adam_beta_1"] ** (epoch + 1))

    ## Setup the V side of the optimization
    layer["layer"].v_w = config["adam_beta_2"] * layer["layer"].v_w + (
        1 - config["adam_beta_2"]
    ) * np.multiply(layer["layer"].L_theta_by_w, layer["layer"].L_theta_by_w)

    v_hat_w = layer["layer"].v_w / (1 - config["adam_beta_2"] ** (epoch + 1))

    layer["layer"].v_b = config["adam_beta_2"] * layer["layer"].v_b + (
        1 - config["adam_beta_2"]
    ) * np.multiply(layer["layer"].L_theta_by_b, layer["layer"].L_theta_by_b)

    v_hat_b = layer["layer"].v_b / (1 - config["adam_beta_2"] ** (epoch + 1))

    updated_weight = np.clip(
        np.multiply(
            (
                config["adam_beta_1"] * m_hat_w
                + (
                    (1 - config["adam_beta_1"])
                    / (1 - config["adam_beta_1"] ** (epoch + 1))
                )
                * layer["layer"].L_theta_by_w
            ),
            (learning_rate / (np.sqrt(v_hat_w) + config["RMS_epsilon"])),
        ),
        a_min=-1,
        a_max=1,
    )
    updated_bias = np.clip(
        np.multiply(
            (
                config["adam_beta_1"] * m_hat_b
                + (
                    (1 - config["adam_beta_1"])
                    / (1 - config["adam_beta_1"] ** (epoch + 1))
                )
                * layer["layer"].L_theta_by_b
            ),
            (learning_rate / (np.sqrt(v_hat_b) + config["RMS_epsilon"])),
        ),
        a_min=-1,
        a_max=1,
    )

    layer["layer"].weight -= updated_weight
    layer["layer"].bias -= updated_bias
