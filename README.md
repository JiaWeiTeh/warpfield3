# WARPFIELD3

<p>A re-implementation of WARPFILEDv2.1</p>

<p>This repo expands on WARPFIELD, and include changes such as:</p>
<ul>
    <li>The ability to include the effect of stochastic sampling of the IMF function in the formation of low-mass clusters.</li>
    <li>Implement an easier-to-use file structure for future users and developers</li>
	<li>Create freedom to use different stellar evolution models (e.g. PARSEC, SB99)</li>
	<li>Improve flexibility in output format (ASCII, FITS, ...)</li>
	<li>Expand metallicity range</li>
</ul>
    
<p>Repo Structure:</p>

<code>./lib</code> Useful libraries which WARPFIELD uses.

<code>./param</code> Folder storing parameter files. Here, users may create a 
<code>.param</code> file following the syntax of <code>example.param</code> to
run WARPFIELD on. 

<code>./src</code> Source code for WARPFIELD.
<ul>
  <li><code>/input_tools</code> useful tools to manipulate inputs</li>
  <li><code>/output_tools</code> useful tools to process outputs</li>
  <li><code>/warpfield</code> contains the main WARPFIELD codes</li>
</ul> 

<p>To run, use the following in your terminal (an example):</p>
<code> python3 -m src.input_tools.run param/example.param</code>