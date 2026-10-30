import logging
import sys

import pandas as pd
from xarray import DataArray

from scripts.prepare_sector_network import determine_emission_sectors

logger = logging.getLogger(__name__)


def add_capacity_limits(n, investment_year, limits_capacity, sense="maximum"):
    for c in n.iterate_components(limits_capacity):
        logger.info(f"Adding {sense} constraints for {c.list_name}")

        attr = "e" if c.name == "Store" else "p"
        units = "MWh or tCO2" if c.name == "Store" else "MW"

        for carrier in limits_capacity[c.name]:
            for ct in limits_capacity[c.name][carrier]:
                if investment_year not in limits_capacity[c.name][carrier][ct].keys():
                    continue

                limit = 1e3 * limits_capacity[c.name][carrier][ct][investment_year]

                logger.info(
                    f"Adding constraint on {c.name} {carrier} capacity in {ct} to be {sense} {limit} {units}"
                )

                valid_components = (
                    (c.static.index.str[:2] == ct)
                    & (c.static.carrier.str[: len(carrier)] == carrier)
                    & ~c.static.carrier.str.contains("thermal")
                )  # exclude solar thermal

                existing_index = c.static.index[
                    valid_components & ~c.static[attr + "_nom_extendable"]
                ]
                extendable_index = c.static.index[
                    valid_components
                    & c.static[attr + "_nom_extendable"]
                    & c.static.active
                ]

                if extendable_index.empty:
                    logger.info(
                        f"No extendable {c.name} with carrier {carrier} found in {ct}. Skipping constraint."
                    )
                    continue

                existing_capacity = c.static.loc[existing_index, attr + "_nom"].sum()

                logger.info(
                    f"Existing {c.name} {carrier} capacity in {ct}: {existing_capacity} {units}"
                )

                nom = n.model[c.name + "-" + attr + "_nom"].loc[extendable_index]

                lhs = nom.sum()

                cname = f"capacity_{sense}-{ct}-{c.name}-{carrier.replace(' ', '-')}"

                if cname in n.global_constraints.index:
                    logger.warning(
                        f"Global constraint {cname} already exists. Dropping and adding it again."
                    )
                    n.global_constraints.drop(cname, inplace=True)

                rhs = limit - existing_capacity

                if sense == "maximum":
                    if rhs <= 0:
                        logger.warning(
                            f"Existing capacity in {ct} for carrier {carrier} already exceeds the limit of {limit} MW. Limiting capacity expansion for this investment period to 0."
                        )
                        rhs = 0

                    n.model.add_constraints(
                        lhs <= rhs,
                        name=f"GlobalConstraint-{cname}",
                    )
                    n.add(
                        "GlobalConstraint",
                        cname,
                        constant=rhs,
                        sense="<=",
                        type="",
                        carrier_attribute="",
                    )

                elif sense == "minimum":
                    n.model.add_constraints(
                        lhs >= rhs,
                        name=f"GlobalConstraint-{cname}",
                    )
                    n.add(
                        "GlobalConstraint",
                        cname,
                        constant=rhs,
                        sense=">=",
                        type="",
                        carrier_attribute="",
                    )
                else:
                    logger.error("sense {sense} not recognised")
                    sys.exit()


def add_power_limits(n, investment_year, limits_power_max):
    """
    " Restricts the maximum inflow/outflow of electricity from/to a country.
    """

    def add_pos_neg_aux_variables(n, idx, var_name, infix):
        """
        For every snapshot in the network `n` this functions adds auxiliary variables corresponding to the positive and negative parts of the dynamical variables of the network components specified in the index `idx`. The `infix` parameter is used to create unique names for the auxiliary variables and constraints.

        Parameters
        ----------
        n : pypsa.Network
            The PyPSA network object containing the model.
        idx : pandas.Index
            The index of the network component (e.g., lines or links) for which to create auxiliary variables.
        infix : str
            A string used to create unique names for the auxiliary variables and constraints.
        """
        var = n.model[var_name].sel({"name": idx})
        aux_pos = n.model.add_variables(
            name=f"{var_name}-{infix}-aux-pos",
            lower=0,
            coords=[n.snapshots, idx],
        )
        aux_neg = n.model.add_variables(
            name=f"{var_name}-{infix}-aux-neg",
            upper=0,
            coords=[n.snapshots, idx],
        )
        n.model.add_constraints(
            aux_pos >= var,
            name=f"{var_name}-{infix}-aux-pos-constr",
        )
        n.model.add_constraints(
            aux_neg <= var,
            name=f"{var_name}-{infix}-aux-neg-constr",
        )
        return aux_pos, aux_neg

    for ct in limits_power_max:
        if investment_year not in limits_power_max[ct].keys():
            continue

        lim = 1e3 * limits_power_max[ct][investment_year]  # in MW

        logger.info(
            f"Adding constraint on electricity import/export from/to {ct} to be < {lim} MW"
        )
        # identify interconnectors

        incoming_lines = n.lines.query(
            f"not bus0.str.startswith('{ct}') and bus1.str.startswith('{ct}') and active"
        )
        outgoing_lines = n.lines.query(
            f"bus0.str.startswith('{ct}') and not bus1.str.startswith('{ct}') and active"
        )
        incoming_links = n.links.query(
            f"not bus0.str.startswith('{ct}') and bus1.str.startswith('{ct}') and carrier == 'DC' and active"
        )
        outgoing_links = n.links.query(
            f"bus0.str.startswith('{ct}') and not bus1.str.startswith('{ct}') and carrier == 'DC' and active"
        )

        # define auxiliary variables for positive and negative parts of line and link flows

        incoming_lines_aux_pos, incoming_lines_aux_neg = add_pos_neg_aux_variables(
            n, incoming_lines.index, "Line-s", f"incoming-{ct}"
        )

        outgoing_lines_aux_pos, outgoing_lines_aux_neg = add_pos_neg_aux_variables(
            n, outgoing_lines.index, "Line-s", f"outgoing-{ct}"
        )

        incoming_links_aux_pos, incoming_links_aux_neg = add_pos_neg_aux_variables(
            n, incoming_links.index, "Link-p", f"incoming-{ct}"
        )

        outgoing_links_aux_pos, outgoing_links_aux_neg = add_pos_neg_aux_variables(
            n, outgoing_links.index, "Link-p", f"outgoing-{ct}"
        )
        # To constraint the absolute values of imports and exports, we have to sum the
        # corresponding positive and negative flows separately, using the auxiliary variables

        import_lhs = (
            incoming_links_aux_pos
            + incoming_lines_aux_pos
            - outgoing_links_aux_neg
            - outgoing_lines_aux_neg
        ).sum(dim="name") / 10

        export_lhs = (
            outgoing_links_aux_pos
            + outgoing_lines_aux_pos
            - incoming_links_aux_neg
            - incoming_lines_aux_neg
        ).sum(dim="name") / 10

        n.model.add_constraints(import_lhs <= lim / 10, name=f"Power-import-limit-{ct}")
        n.model.add_constraints(export_lhs <= lim / 10, name=f"Power-export-limit-{ct}")


def h2_import_limits(n, investment_year, limits_volume_max):
    for ct in limits_volume_max["h2_import"]:
        limit = limits_volume_max["h2_import"][ct][investment_year] * 1e6

        logger.info(f"limiting H2 imports in {ct} to {limit / 1e6} TWh/a")
        pipeline_carrier = [
            "H2 pipeline",
            "H2 pipeline (Kernnetz)",
            "H2 pipeline retrofitted",
        ]
        incoming = n.links.index[
            (n.links.carrier.isin(pipeline_carrier))
            & (n.links.bus0.str[:2] != ct)
            & (n.links.bus1.str[:2] == ct)
        ]
        outgoing = n.links.index[
            (n.links.carrier.isin(pipeline_carrier))
            & (n.links.bus0.str[:2] == ct)
            & (n.links.bus1.str[:2] != ct)
        ]

        incoming_p = (
            n.model["Link-p"].loc[:, incoming] * n.snapshot_weightings.generators
        ).sum()
        outgoing_p = (
            n.model["Link-p"].loc[:, outgoing] * n.snapshot_weightings.generators
        ).sum()

        lhs = incoming_p - outgoing_p

        cname = f"H2_import_limit-{ct}"

        n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )

        logger.info("Adding H2 export ban")

        cname = f"H2_export_ban-{ct}"

        n.model.add_constraints(lhs >= 0, name=f"GlobalConstraint-{cname}")

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=0,
            sense=">=",
            type="",
            carrier_attribute="",
        )


def h2_production_limits(n, investment_year, limits_volume_min, limits_volume_max):
    for ct in limits_volume_max["electrolysis"]:
        if ct not in limits_volume_min["electrolysis"]:
            logger.warning(
                f"no lower limit for H2 electrolysis in {ct} assuming 0 TWh/a"
            )
            limit_lower = 0
        else:
            limit_lower = limits_volume_min["electrolysis"][ct][investment_year] * 1e6

        limit_upper = limits_volume_max["electrolysis"][ct][investment_year] * 1e6

        logger.info(
            f"limiting H2 electrolysis in DE between {limit_lower / 1e6} and {limit_upper / 1e6} TWh/a"
        )

        production = n.links[
            (n.links.carrier == "H2 Electrolysis") & (n.links.bus0.str.contains(ct))
        ].index
        efficiency = n.links.loc[production, "efficiency"]

        lhs = (
            n.model["Link-p"].loc[:, production]
            * n.snapshot_weightings.generators
            * efficiency
        ).sum()

        cname_upper = f"H2_production_limit_upper-{ct}"
        cname_lower = f"H2_production_limit_lower-{ct}"

        n.model.add_constraints(
            lhs <= limit_upper, name=f"GlobalConstraint-{cname_upper}"
        )

        n.model.add_constraints(
            lhs >= limit_lower, name=f"GlobalConstraint-{cname_lower}"
        )

        if cname_upper not in n.global_constraints.index:
            n.add(
                "GlobalConstraint",
                cname_upper,
                constant=limit_upper,
                sense="<=",
                type="",
                carrier_attribute="",
            )
        if cname_lower not in n.global_constraints.index:
            n.add(
                "GlobalConstraint",
                cname_lower,
                constant=limit_lower,
                sense=">=",
                type="",
                carrier_attribute="",
            )


def electricity_import_limits(n, investment_year, limits_volume_max):
    for ct in limits_volume_max["electricity_import"]:
        limit = limits_volume_max["electricity_import"][ct][investment_year] * 1e6

        if limit < 0:
            limit *= n.snapshot_weightings.generators.sum() / 8760

        logger.info(f"limiting electricity imports in {ct} to {limit / 1e6} TWh/a")

        incoming_line = n.lines.index[
            (n.lines.carrier == "AC")
            & (n.lines.bus0.str[:2] != ct)
            & (n.lines.bus1.str[:2] == ct)
        ]
        outgoing_line = n.lines.index[
            (n.lines.carrier == "AC")
            & (n.lines.bus0.str[:2] == ct)
            & (n.lines.bus1.str[:2] != ct)
        ]

        incoming_link = n.links.index[
            (n.links.carrier == "DC")
            & (n.links.bus0.str[:2] != ct)
            & (n.links.bus1.str[:2] == ct)
        ]
        outgoing_link = n.links.index[
            (n.links.carrier == "DC")
            & (n.links.bus0.str[:2] == ct)
            & (n.links.bus1.str[:2] != ct)
        ]

        incoming_line_p = (
            n.model["Line-s"].loc[:, incoming_line] * n.snapshot_weightings.generators
        ).sum()
        outgoing_line_p = (
            n.model["Line-s"].loc[:, outgoing_line] * n.snapshot_weightings.generators
        ).sum()

        incoming_link_p = (
            n.model["Link-p"].loc[:, incoming_link] * n.snapshot_weightings.generators
        ).sum()
        outgoing_link_p = (
            n.model["Link-p"].loc[:, outgoing_link] * n.snapshot_weightings.generators
        ).sum()

        lhs = (incoming_link_p - outgoing_link_p) + (incoming_line_p - outgoing_line_p)

        cname = f"Electricity_import_limit-{ct}"

        n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )


def add_national_co2_budgets(n, snakemake, national_co2_budgets, investment_year):
    """
    Add a set of emissions limit constraints for specified countries.

    The countries and emissions limits are specified in the config file entry 'co2_budget_national'.

    Parameters
    ----------
    n : pypsa.Network
    snakemake : snakemake.io.Snakemake
    national_co2_budgets : dict
    investment_year : int

    """
    logger.info("Adding national CO2 budgets")
    nhours = n.snapshot_weightings.generators.sum()
    nyears = nhours / 8760

    sectors = determine_emission_sectors(n.config["sector"])
    energy_totals = pd.read_csv(snakemake.input.energy_totals, index_col=[0, 1])

    # convert MtCO2 to tCO2
    co2_totals = 1e6 * pd.read_csv(snakemake.input.co2_totals_name, index_col=0)

    co2_total_totals = co2_totals[sectors].sum(axis=1) * nyears

    for ct in national_co2_budgets:
        if ct != "DE":
            logger.error(
                f"CO2 budget for countries other than `DE` is not yet supported. Found country {ct}. Please check the config file."
            )

        limit = co2_total_totals[ct] * national_co2_budgets[ct][investment_year]
        logger.info(
            f"Limiting emissions in country {ct} to {national_co2_budgets[ct][investment_year]:.1%} of "
            f"1990 levels, i.e. {limit:,.2f} tCO2/a",
        )

        lhs = []

        for port in [col[3:] for col in n.links if col.startswith("bus")]:
            links = n.links.index[
                (n.links.index.str[:2] == ct)
                & (n.links[f"bus{port}"] == "co2 atmosphere")
                & ~n.links.carrier.str.contains(
                    "shipping|aviation"
                )  # first exclude aviation to multiply it with a domestic factor later
            ]

            logger.info(
                f"For {ct} adding following link carriers to port {port} CO2 constraint: {n.links.loc[links, 'carrier'].unique()}"
            )

            if port == "0":
                efficiency = -1.0
            elif port == "1":
                efficiency = n.links.loc[links, "efficiency"]
            else:
                efficiency = n.links.loc[links, f"efficiency{port}"]

            lhs.append(
                (
                    n.model["Link-p"].loc[:, links]
                    * efficiency
                    * n.snapshot_weightings.generators
                ).sum()
            )

        # Aviation demand
        domestic_aviation = energy_totals.loc[
            (ct, snakemake.params.energy_year), "total domestic aviation"
        ]
        international_aviation = energy_totals.loc[
            (ct, snakemake.params.energy_year), "total international aviation"
        ]
        domestic_aviation_factor = domestic_aviation / (
            domestic_aviation + international_aviation
        )
        aviation_links = n.links[
            (n.links.index.str[:2] == ct) & (n.links.carrier == "kerosene for aviation")
        ]
        lhs.append(
            (
                n.model["Link-p"].loc[:, aviation_links.index]
                * aviation_links.efficiency2
                * n.snapshot_weightings.generators
            ).sum()
            * domestic_aviation_factor
        )
        logger.info(
            f"Adding domestic aviation emissions for {ct} with a factor of {domestic_aviation_factor}"
        )

        # Shipping oil
        domestic_navigation = energy_totals.loc[
            (ct, snakemake.params.energy_year), "total domestic navigation"
        ]
        international_navigation = energy_totals.loc[
            (ct, snakemake.params.energy_year), "total international navigation"
        ]
        domestic_navigation_factor = domestic_navigation / (
            domestic_navigation + international_navigation
        )
        shipping_links = n.links[
            (n.links.index.str[:2] == ct) & (n.links.carrier == "shipping oil")
        ]
        lhs.append(
            (
                n.model["Link-p"].loc[:, shipping_links.index]
                * shipping_links.efficiency2
                * n.snapshot_weightings.generators
            ).sum()
            * domestic_navigation_factor
        )

        # Shipping methanol
        shipping_meoh_links = n.links[
            (n.links.index.str[:2] == ct) & (n.links.carrier == "shipping methanol")
        ]
        if not shipping_meoh_links.empty:  # no shipping methanol in 2025
            lhs.append(
                (
                    n.model["Link-p"].loc[:, shipping_meoh_links.index]
                    * shipping_meoh_links.efficiency2
                    * n.snapshot_weightings.generators
                ).sum()
                * domestic_navigation_factor
            )

        logger.info(
            f"Adding domestic shipping emissions for {ct} with a factor of {domestic_navigation_factor}"
        )

        # Adding Efuel imports and exports to constraint
        incoming_oil = n.links.index[n.links.index == f"EU renewable oil -> {ct} oil"]
        outgoing_oil = n.links.index[n.links.index == f"{ct} renewable oil -> EU oil"]

        lhs.append(
            (
                -1
                * n.model["Link-p"].loc[:, incoming_oil]
                * 0.2571
                * n.snapshot_weightings.generators
            ).sum()
        )
        lhs.append(
            (
                n.model["Link-p"].loc[:, outgoing_oil]
                * 0.2571
                * n.snapshot_weightings.generators
            ).sum()
        )

        incoming_methanol = n.links.index[
            n.links.index == f"EU methanol -> {ct} methanol"
        ]
        outgoing_methanol = n.links.index[
            n.links.index == f"{ct} methanol -> EU methanol"
        ]

        methanol_emissions = n.links.loc["EU industry methanol", "efficiency2"]
        lhs.append(
            (
                -1
                * n.model["Link-p"].loc[:, incoming_methanol]
                * methanol_emissions
                * n.snapshot_weightings.generators
            ).sum()
        )

        lhs.append(
            (
                n.model["Link-p"].loc[:, outgoing_methanol]
                * methanol_emissions
                * n.snapshot_weightings.generators
            ).sum()
        )

        # Methane
        incoming_CH4 = n.links.index[n.links.index == f"EU renewable gas -> {ct} gas"]
        outgoing_CH4 = n.links.index[n.links.index == f"{ct} renewable gas -> EU gas"]

        lhs.append(
            (
                -1
                * n.model["Link-p"].loc[:, incoming_CH4]
                * 0.198
                * n.snapshot_weightings.generators
            ).sum()
        )

        lhs.append(
            (
                n.model["Link-p"].loc[:, outgoing_CH4]
                * 0.198
                * n.snapshot_weightings.generators
            ).sum()
        )

        lhs = sum(lhs)

        cname = f"co2_limit-{ct}"

        n.model.add_constraints(
            lhs <= limit,
            name=f"GlobalConstraint-{cname}",
        )

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )


def force_boiler_profiles_existing_per_load(n):
    """
    This scales the boiler dispatch to the load profile with a factor common to
    all boilers at load.
    """

    logger.info("Forcing boiler profiles for existing ones")

    decentral_boilers = n.links.index[
        n.links.carrier.str.contains("boiler")
        & ~n.links.carrier.str.contains("urban central")
        & ~n.links.p_nom_extendable
    ]

    if decentral_boilers.empty:
        return

    boiler_loads = n.links.loc[decentral_boilers, "bus1"]
    boiler_loads = boiler_loads[boiler_loads.isin(n.loads_t.p_set.columns)]
    decentral_boilers = boiler_loads.index
    boiler_profiles_pu = n.loads_t.p_set[boiler_loads].div(
        n.loads_t.p_set[boiler_loads].max(), axis=1
    )
    boiler_profiles_pu.columns = decentral_boilers
    boiler_profiles = DataArray(
        boiler_profiles_pu.multiply(n.links.loc[decentral_boilers, "p_nom"], axis=1)
    )

    boiler_load_index = pd.Index(boiler_loads.unique())
    boiler_load_index.name = "Load"

    # per load scaling factor
    n.model.add_variables(coords=[boiler_load_index], name="Load-profile_factor")

    # clumsy indicator matrix to map boilers to loads
    df = pd.DataFrame(index=boiler_load_index, columns=decentral_boilers, data=0.0)
    for k, v in boiler_loads.items():
        df.loc[v, k] = 1.0

    lhs = n.model["Link-p"].loc[:, decentral_boilers] - (
        boiler_profiles * DataArray(df) * n.model["Load-profile_factor"]
    ).sum("Load")

    n.model.add_constraints(lhs, "=", 0, "Link-fixed_profile")

    # hack so that PyPSA doesn't complain there is nowhere to store the variable
    n.loads["profile_factor_opt"] = 0.0


def force_boiler_profiles_existing_per_boiler(n):
    """
    This scales each boiler dispatch to be proportional to the load profile.
    """

    logger.info(
        "Forcing each existing boiler dispatch to be proportional to the load profile"
    )

    decentral_boilers = n.links.index[
        n.links.carrier.str.contains("boiler")
        & ~n.links.carrier.str.contains("urban central")
        & ~n.links.p_nom_extendable
    ]

    if decentral_boilers.empty:
        return

    boiler_loads = n.links.loc[decentral_boilers, "bus1"]
    boiler_loads = boiler_loads[boiler_loads.isin(n.loads_t.p_set.columns)]
    decentral_boilers = boiler_loads.index
    boiler_profiles_pu = n.loads_t.p_set[boiler_loads].div(
        n.loads_t.p_set[boiler_loads].max(), axis=1
    )
    boiler_profiles_pu.columns = decentral_boilers
    boiler_profiles = DataArray(
        boiler_profiles_pu.multiply(n.links.loc[decentral_boilers, "p_nom"], axis=1)
    )

    # will be per unit
    n.model.add_variables(coords=[decentral_boilers], name="Link-fixed_profile_scaling")

    lhs = (
        (1, n.model["Link-p"].loc[:, decentral_boilers]),
        (
            -boiler_profiles,
            n.model["Link-fixed_profile_scaling"],
        ),
    )

    n.model.add_constraints(lhs, "=", 0, "Link-fixed_profile_scaling")

    # hack so that PyPSA doesn't complain there is nowhere to store the variable
    n.links["fixed_profile_scaling_opt"] = 0.0


def add_h2_derivate_limit(n, investment_year, limits_volume_max):
    for ct in limits_volume_max["h2_derivate_import"]:
        limit = limits_volume_max["h2_derivate_import"][ct][investment_year] * 1e6

        logger.info(f"limiting H2 derivate imports in {ct} to {limit / 1e6} TWh/a")

        incoming = n.links.loc[
            [
                "EU renewable oil -> DE oil",
                "EU methanol -> DE methanol",
                "EU renewable gas -> DE gas",
            ]
        ].index
        outgoing = n.links.loc[
            [
                "DE renewable oil -> EU oil",
                "DE methanol -> EU methanol",
                "DE renewable gas -> EU gas",
            ]
        ].index

        carrier_idx_dict = {
            # Every carrier should respect the limit individually
            "renewable_oil": 0,
            "methanol": 1,
            "renewable_gas": 2,
            # Exports of one carrier should not compensate for imports of another carrier
            "H2_derivate_oil_meoh": [0, 1],
            "H2_derivate_oil_gas": [0, 2],
            "H2_derivate_meoh_gas": [1, 2],
            # The sum of all carriers should respect the limit
            "H2_derivate_oil_meoh_gas": [0, 1, 2],
        }
        for carrier, idx in carrier_idx_dict.items():
            cname = f"{carrier}_import_limit-{ct}"

            incoming_p = (
                n.model["Link-p"].loc[:, incoming[idx]]
                * n.snapshot_weightings.generators
            ).sum()
            outgoing_p = (
                n.model["Link-p"].loc[:, outgoing[idx]]
                * n.snapshot_weightings.generators
            ).sum()

            lhs = incoming_p - outgoing_p

            n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

            if cname in n.global_constraints.index:
                logger.warning(
                    f"Global constraint {cname} already exists. Dropping and adding it again."
                )
                n.global_constraints.drop(cname, inplace=True)

            n.add(
                "GlobalConstraint",
                cname,
                constant=limit,
                sense="<=",
                type="",
                carrier_attribute="",
            )

    # Export bans on efuels are implemented in modify_prenetwork by restricting p_max_pu of the DE -> EU links


def adapt_nuclear_output(n):
    logger.info(
        "limiting german electricity generation from nuclear to 2020 value of 61 TWh"
    )
    limit = 61e6

    nuclear_de_index = n.links.index[
        (n.links.carrier == "nuclear") & (n.links.index.str[:2] == "DE")
    ]

    nuclear_gen = (
        n.model["Link-p"].loc[:, nuclear_de_index]
        * n.links.loc[nuclear_de_index, "efficiency"]
        * n.snapshot_weightings.generators
    ).sum()

    lhs = nuclear_gen

    cname = "Nuclear_generation_limit-DE"

    n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

    if cname in n.global_constraints.index:
        logger.warning(
            f"Global constraint {cname} already exists. Dropping and adding it again."
        )
        n.global_constraints.drop(cname, inplace=True)

    n.add(
        "GlobalConstraint",
        cname,
        constant=limit,
        sense="<=",
        type="",
        carrier_attribute="",
    )

####### Anpassung LuKa #############

def add_sr27_h2_capacity(n, investment_year, targets_gw):
    """Aktive deutsche H2-Kraftwerksleistung auf Zielwert in GW_el setzen."""
    target = targets_gw.get(
        investment_year, targets_gw.get(str(investment_year))
    )
    if target is None:
        return

    target_mw = float(target) * 1e3
    if not 0 <= target_mw < float("inf"):
        raise ValueError("H2-Ziel muss endlich und nicht negativ sein.")

    links = n.links
    h2_carriers = [
        "H2 OCGT",
        "H2 CCGT",
        "H2 retrofit OCGT",
        "H2 retrofit CCGT",
        "urban central H2 CHP",
        "urban central H2 retrofit CHP",
    ]

    selected = (
        links.carrier.isin(h2_carriers)
        & links.bus1.str.startswith("DE", na=False)
        & links.bus1.map(n.buses.carrier).eq("AC")
        & links.active
    )

    efficiency = links.loc[selected, "efficiency"]
    if (
        efficiency.isna().any()
        or (~efficiency.gt(0)).any()
        or (~efficiency.lt(float("inf"))).any()
    ):
        raise ValueError("H2-Kraftwerke haben ungültige Wirkungsgrade.")

    fixed = selected & ~links.p_nom_extendable
    extendable = links.index[selected & links.p_nom_extendable]

    existing_mw = (
        links.loc[fixed, "p_nom"] * links.loc[fixed, "efficiency"]
    ).sum()
    remaining_mw = target_mw - existing_mw

    if remaining_mw < -1e-3:
        raise ValueError(
            f"SR27 {investment_year}: fester H2-Bestand "
            f"{existing_mw / 1e3:.3f} GW_el übersteigt "
            f"Ziel {target_mw / 1e3:.3f} GW_el."
        )

    if extendable.empty:
        if abs(remaining_mw) > 1e-3:
            raise ValueError(
                f"SR27 {investment_year}: H2-Ziel nicht erreichbar, "
                "keine ausbaubaren H2-Kraftwerke vorhanden."
            )
        return

    # Link-p_nom ist Eingangsleistung; Wirkungsgrad liefert MW_el.
    nom = n.model["Link-p_nom"].loc[extendable]
    eta = links.loc[extendable, "efficiency"].to_numpy()
    lhs = (nom * eta).sum()

    n.model.add_constraints(
        lhs == max(remaining_mw, 0.0),
        name=f"SR27-H2-capacity-DE-{investment_year}",
    )

    logger.info(
        f"SR27 {investment_year}: H2-Gesamtziel "
        f"{target_mw / 1e3:.3f} GW_el, davon "
        f"{existing_mw / 1e3:.3f} GW_el fester Bestand."
    )

def add_sr27_methane_capacity(n, investment_year, targets_gw):
    """Aktive deutsche Methankraftwerksleistung auf Ziel in GW_el setzen."""
    target = targets_gw.get(
        investment_year, targets_gw.get(str(investment_year))
    )
    if target is None:
        return

    target_mw = float(target) * 1e3
    if not 0 <= target_mw < float("inf"):
        raise ValueError(
            "Methan-Gesamtziel muss endlich und nicht negativ sein."
        )

    links = n.links
    gas_carriers = [
        "OCGT",
        "CCGT",
        "urban central gas CHP",
        "urban central gas CHP CC",
    ]

    selected = (
        links.carrier.isin(gas_carriers)
        & links.bus1.str.startswith("DE", na=False)
        & links.bus1.map(n.buses.carrier).eq("AC")
        & links.active
    )

    efficiency = links.loc[selected, "efficiency"]
    if (
        efficiency.isna().any()
        or (~efficiency.gt(0)).any()
        or (~efficiency.lt(float("inf"))).any()
    ):
        raise ValueError(
            "Methankraftwerke haben ungueltige Wirkungsgrade."
        )

    fixed = selected & ~links.p_nom_extendable
    extendable = links.index[selected & links.p_nom_extendable]

    existing_mw = (
        links.loc[fixed, "p_nom"] * links.loc[fixed, "efficiency"]
    ).sum()
    remaining_mw = target_mw - existing_mw

    if remaining_mw < -1e-3:
        raise ValueError(
            f"SR27 {investment_year}: fester Methanbestand "
            f"{existing_mw / 1e3:.3f} GW_el uebersteigt "
            f"Gesamtziel {target_mw / 1e3:.3f} GW_el."
        )

    if extendable.empty:
        if abs(remaining_mw) > 1e-3:
            raise ValueError(
                f"SR27 {investment_year}: Methan-Gesamtziel "
                "nicht erreichbar, keine ausbaubaren "
                "Methankraftwerke vorhanden."
            )

        logger.info(
            f"SR27 {investment_year}: Methan-Gesamtziel bereits "
            f"durch festen Bestand erfuellt: "
            f"{existing_mw / 1e3:.3f} GW_el."
        )
        return

    # Link-p_nom ist Eingangsleistung; Umrechnung in MW_el.
    nom = n.model["Link-p_nom"].loc[extendable]
    eta = links.loc[extendable, "efficiency"].to_numpy()
    lhs = (nom * eta).sum()

    n.model.add_constraints(
        lhs == max(remaining_mw, 0.0),
        name=f"SR27-methane-capacity-DE-{investment_year}",
    )

    logger.info(
        f"SR27 {investment_year}: Methan-Gesamtziel "
        f"{target_mw / 1e3:.3f} GW_el, davon "
        f"{existing_mw / 1e3:.3f} GW_el fester Bestand."
    )


def add_sr27_biomethane(n, investment_year, targets_twh):
    """Biomethan-Einspeisung aus deutscher Aufbereitung auf Ziel in TWh setzen."""
    target = targets_twh.get(investment_year, targets_twh.get(str(investment_year)))
    if target is None:
        return

    target_mwh = float(target) * 1e6
    if not 0 <= target_mwh < float("inf"):
        raise ValueError("Biomethan-Ziel muss endlich und nicht negativ sein.")

    links = n.links
    links_i = links.index[
        links.carrier.isin(["biogas to gas", "biogas to gas CC"])
        & links.bus0.str.startswith("DE", na=False)
        & links.active
    ]
    if links_i.empty:
        raise ValueError(
            f"SR27 {investment_year}: keine deutschen Biogas-Aufbereitungen gefunden."
        )

    # Link-p ist Biogas-Eingang; Gewichtung x Wirkungsgrad ergibt MWh Biomethan.
    p = n.model["Link-p"].loc[:, links_i]
    weights = n.snapshot_weightings.generators.loc[n.snapshots].to_numpy()
    efficiency = links.loc[links_i, "efficiency"].to_numpy()
    coeff = DataArray(
        weights[:, None] * efficiency[None, :],
        coords=[p.coords[dim] for dim in p.dims],
    )
    lhs = (p * coeff).sum()

    n.model.add_constraints(
        lhs == target_mwh,
        name=f"SR27-biomethane-DE-{investment_year}",
    )

    logger.info(
        f"SR27 {investment_year}: Biomethan-Einspeisung DE auf "
        f"{target_mwh / 1e6:.2f} TWh/a gesetzt."
    )


def add_sr27_building_gas(n, investment_year, targets_twh):
    """Gaseinsatz dezentraler Kessel in DE (Haushalte + GHD) auf Ziel in TWh setzen."""
    target = targets_twh.get(investment_year, targets_twh.get(str(investment_year)))
    if target is None:
        return

    target_mwh = float(target) * 1e6
    if not 0 <= target_mwh < float("inf"):
        raise ValueError("Gebäude-Gasziel muss endlich und nicht negativ sein.")

    links = n.links
    links_i = links.index[
        links.carrier.str.contains("gas boiler")
        & ~links.carrier.str.contains("urban central")
        & links.index.str.startswith("DE")
        & links.active
    ]
    if links_i.empty:
        raise ValueError(
            f"SR27 {investment_year}: keine deutschen dezentralen Gaskessel gefunden; "
            "Kesselverbot (new_decentral_fossil_boiler_ban) prüfen."
        )

    # Link-p ist der Gaseinsatz (Heizwert); die SR-Mengen sind Gas-Ausspeisung.
    p = n.model["Link-p"].loc[:, links_i]
    weights = n.snapshot_weightings.generators.loc[n.snapshots]
    lhs = (p * weights).sum()

    n.model.add_constraints(
        lhs == target_mwh,
        name=f"SR27-building-gas-DE-{investment_year}",
    )

    logger.info(
        f"SR27 {investment_year}: Gaseinsatz dezentraler Kessel DE auf "
        f"{target_mwh / 1e6:.2f} TWh/a gesetzt ({len(links_i)} Links)."
    )
###########################################################################################

def additional_functionality(n, snapshots, snakemake):
    logger.info("Adding Ariadne-specific functionality")

    investment_year = int(snakemake.wildcards.planning_horizons[-4:])
    constraints = snakemake.params.solving["constraints"]

    add_capacity_limits(
        n, investment_year, constraints["limits_capacity_min"], "minimum"
    )

    add_capacity_limits(
        n, investment_year, constraints["limits_capacity_max"], "maximum"
    )

    # Anpassung LuKa: SR27-H2-Summenbedingung aufrufen
    add_sr27_h2_capacity(
        n,
        investment_year,
        constraints.get("sr27_h2_capacity_gw", {}),
    )

    add_sr27_methane_capacity(
        n,
        investment_year,
        constraints.get("sr27_methane_capacity_gw", {}),
    )

    add_sr27_biomethane(
        n,
        investment_year,
        constraints.get("sr27_biomethane_twh", {}),
    )

    add_sr27_building_gas(
        n,
        investment_year,
        constraints.get("sr27_building_gas_twh", {}),
    )
    ############

    add_power_limits(n, investment_year, constraints["limits_power_max"])

    if snakemake.wildcards.clusters != "1":
        h2_import_limits(n, investment_year, constraints["limits_volume_max"])

        electricity_import_limits(n, investment_year, constraints["limits_volume_max"])

    if investment_year >= 2025:
        h2_production_limits(
            n,
            investment_year,
            constraints["limits_volume_min"],
            constraints["limits_volume_max"],
        )

    add_h2_derivate_limit(n, investment_year, constraints["limits_volume_max"])

    # force_boiler_profiles_existing_per_load(n)
    force_boiler_profiles_existing_per_boiler(n)

    if isinstance(constraints["co2_budget_national"], dict):
        add_national_co2_budgets(
            n,
            snakemake,
            constraints["co2_budget_national"],
            investment_year,
        )
    else:
        logger.warning("No national CO2 budget specified!")

    if investment_year == 2020:
        adapt_nuclear_output(n)