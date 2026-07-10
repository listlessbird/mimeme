import { describe, expect, it } from "vitest";

import { Route as ImagesRoute } from "./images/index";
import { Route as IngestionRoute } from "./ingestion/index";
import { Route as SourcesRoute } from "./sources/index";

describe("admin section error boundaries", () => {
	it.each([
		["sources", SourcesRoute],
		["images", ImagesRoute],
		["ingestion", IngestionRoute],
	])("keeps a %s API failure inside its child route", (_name, route) => {
		expect(route.options.errorComponent).toBeTypeOf("function");
	});
});
