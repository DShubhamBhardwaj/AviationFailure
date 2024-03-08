Our goal is to develop a warning system that informs aerospace engineers in good time about
impending aircraft breakdowns and offers enough time to prepare for the corresponding
maintenance measures. The purpose of the project is to use predictive analysis for forecasting
failures in aircrafts which will ensure the safety of passengers travelling through that aircraft/
airline.

In this project we show how damage propagation can be modeled within the modules of aircraft
gas turbine engines. To that end, response surfaces of all sensors are generated via a thermo-dynamical
simulation model for the engine as a function of variations of flow and efficiency of the modules of
interest. An exponential rate of change for flow and efficiency loss was imposed for each data set,
starting at a randomly chosen initial deterioration set point. The rate of change of the flow and
efficiency denotes an otherwise unspecified fault with increasingly worsening effect. The rates of change
of the faults were constrained to an upper threshold but were otherwise chosen randomly. Damage
propagation was allowed to continue until a failure criterion was reached. A health index was defined as
the minimum of several superimposed operational margins at any given time instant and the failure
criterion is reached when health index reaches zero. Output of the model was the time series (cycles) of
sensed measurements typically available from aircraft gas turbine engines.

In the project we have NASA turbo fan jet engine dataset that has the readings of 26 sensors in
total and has the details of remaining useful life of the turbo engines based on the sensor reading. We
plan on making a prediction model that would compare the prediction done by various algorithms and
would predict the remaining life of an engine, i.e. the number of hours/Cycles It can fly without any
hinderance/ risk of failure. Once the remaining Cycles are completed the System recommends the
engineering team to keep proper maintenance as the engine becomes prone to failure in such cases.
